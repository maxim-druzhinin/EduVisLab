"""
Флаг «Логик» — качество речевого канала.

Компоненты:
  1. Технические метрики звука (из audio_quality — уже вычислены)
  2. LLM-анализ транскрипта: слова-паразиты, просторечия, орфоэпия
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field, asdict

from openai import OpenAI

logger = logging.getLogger(__name__)

# ─── Thresholds ───────────────────────────────────────────────────────────────

AUDIO_SCORE_THRESHOLD = 5.0

# ─── LLM ──────────────────────────────────────────────────────────────────────
from config import DEEPSEEK_API_KEY, NARRATIVE_LLM_MODEL as DEEPSEEK_MODEL
 
DEEPSEEK_BASE_URL = "https://api.deepseek.com"

LOGIC_LLM_PROMPT = r"""
Ты оцениваешь речевую культуру лектора в YouTube-лекции.

КОНТЕКСТ. Это устная речь живого человека, не литературная норма и не диктант.
У лектора есть право на разговорный стиль, авторскую интонацию, структурные
маркеры речи («так», «вот», «итак») и редкие оговорки. Флаг должен срабатывать
только когда речь продолжительно и навязчиво мешает воспринимать материал,
а не за отдельные шероховатости.

Ответь ТОЛЬКО валидным JSON без markdown:

{
  "fillers": {
    "pattern": "clean | natural | excessive",
    "detected": true/false,
    "examples": ["цитата из транскрипта", "..."]
  },
  "non_literary": {
    "pattern": "clean | natural | excessive",
    "detected": true/false,
    "examples": ["цитата из транскрипта", "..."]
  }
}

Что оцениваем:

- fillers: слова-паразиты — затычки в речи, которые не несут смысла и служат
  заполнителями пауз («ну», «как бы», «короче», «это самое» и подобные).
  Сами по себе они не являются дефектом — вопрос в том, идут ли они с такой
  плотностью, что речь становится рваной и внимание цепляется за повторы.

- non_literary: ненормативная речь, неуместная в образовательном контексте —
  просторечные формы слов («ихний», «ложить» и подобные), грубые ошибки
  согласования и словоупотребления, аграмматичные конструкции. Не путать
  с разговорным стилем — он допустим; речь о словах и формах, которые
  относятся к сниженному регистру или нарушают грамматическую норму.

Шкала pattern (общая для обоих критериев):

- clean — паразитов / нелитературных форм нет, либо они теряются в речи
  и не образуют устойчивого паттерна.

- natural — есть характерные словечки, но они органично встроены в живую
  подачу и не выбиваются из общего регистра. Это часть авторского стиля,
  а не дефект — слушать не мешает.

- excessive — повторяются настолько часто, что речь становится рваной
  и внимание цепляется за повторы. Читается как устойчивая привычка
  лектора, а не как стиль.

ПРАВИЛО: detected = true только при pattern = "excessive".

В examples — конкретные короткие цитаты из транскрипта (1–3 штуки),
демонстрирующие найденный паттерн. Без них вердикт непроверяем.
Пустой список, если pattern = "clean".

НЕ засчитывай: структурные маркеры речи и связки между мыслями, общий
разговорный регистр подачи, технический жаргон, авторские интонации
и апелляции к зрителю, единичные оговорки и самокоррекции, артефакты
whisper-сегментации (обрывы фраз, повторы на границах сегментов).
"""


# ─── Data structures ──────────────────────────────────────────────────────────

@dataclass
class LogicAudioResult:
    flag: bool
    ovrl_mos: float
    sig_mos:  float
    bak_mos:  float
    snr_db:   float
    lufs:     float
    clipping: str
    triggered_by: list[str] = field(default_factory=list)


@dataclass
class LogicLLMResult:
    flag: bool
    fillers_detected: bool
    colloquialisms_detected: bool
    orthoepic_errors_detected: bool
    examples: dict = field(default_factory=dict)
    confidence: float = 0.0
    error: str | None = None


@dataclass
class LogicFlagResult:
    flag: bool
    confidence: float
    audio: dict
    speech_quality: dict
    triggered_by: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


# ─── LLM speech quality check ─────────────────────────────────────────────────

def _check_speech_quality(transcript_text: str) -> LogicLLMResult:
    """LLM-анализ транскрипта на паразиты, просторечия, орфоэпию."""

    if not DEEPSEEK_API_KEY:
        logger.warning("DEEPSEEK_API_KEY не задан — LLM-анализ пропущен")
        return LogicLLMResult(flag=False, fillers_detected=False,
                              colloquialisms_detected=False,
                              orthoepic_errors_detected=False,
                              error="no_api_key")

    client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)

    # Ограничиваем транскрипт чтобы не превысить контекст
    text_sample = transcript_text

    try:
        resp = client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            temperature=0.0,
            messages=[
                {"role": "system", "content": LOGIC_LLM_PROMPT},
                {"role": "user",   "content": f"Транскрипт лекции:\n\n{text_sample}"},
            ],
        )
        raw = resp.choices[0].message.content.strip()
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        logger.error(f"LLM вернул невалидный JSON: {e}")
        return LogicLLMResult(flag=False, fillers_detected=False,
                              colloquialisms_detected=False,
                              orthoepic_errors_detected=False,
                              error=f"json_parse_error: {e}")
    except Exception as e:
        logger.error(f"Ошибка LLM-запроса: {e}")
        return LogicLLMResult(flag=False, fillers_detected=False,
                              colloquialisms_detected=False,
                              orthoepic_errors_detected=False,
                              error=str(e))

    fillers   = data.get("fillers",   {}).get("detected", False)
    colloqui  = data.get("colloquialisms", {}).get("detected", False)
    orthoep   = data.get("orthoepic_errors", {}).get("detected", False)

    return LogicLLMResult(
        flag=fillers or colloqui or orthoep,
        fillers_detected=fillers,
        colloquialisms_detected=colloqui,
        orthoepic_errors_detected=orthoep,
        examples={
            "fillers":    data.get("fillers",   {}).get("examples", []),
            "colloquialisms": data.get("colloquialisms", {}).get("examples", []),
            "orthoepic_errors": data.get("orthoepic_errors", {}).get("examples", []),
        },
        confidence=float(data.get("confidence", 0.5)),
    )


# ─── Main entry point ─────────────────────────────────────────────────────────

def run(
    audio_score: float,
    transcript_text: str,
) -> LogicFlagResult:

    logger.info("── Флаг Логик: проверка качества звука ──")


    audio_flag = audio_score < AUDIO_SCORE_THRESHOLD

    triggered = []

    if audio_flag:
        triggered.append(
            f"audio_score={audio_score:.2f} < {AUDIO_SCORE_THRESHOLD}"
        )

    logger.info("── Флаг Логик: LLM-анализ речи ──")

    llm_result = _check_speech_quality(transcript_text)

    if llm_result.flag and not llm_result.error:

        if llm_result.fillers_detected:
            triggered.append("fillers")

        if llm_result.colloquialisms_detected:
            triggered.append("colloquialisms")

        if llm_result.orthoepic_errors_detected:
            triggered.append("orthoepic_errors")

    flag = (
        audio_flag
        or (
            llm_result.flag
            and not llm_result.error
        )
    )

    if audio_flag:
        confidence = 0.95

    elif llm_result.flag:
        confidence = llm_result.confidence

    else:
        confidence = max(
            0.70,
            llm_result.confidence,
        )

    return LogicFlagResult(
        flag=flag,
        confidence=round(confidence, 3),
        audio={
            "flag": audio_flag,
            "score": round(audio_score, 3),
            "threshold": AUDIO_SCORE_THRESHOLD,
            "triggered_by": (
                [
                    f"audio_score={audio_score:.2f} < {AUDIO_SCORE_THRESHOLD}"
                ]
                if audio_flag
                else []
            ),
        },
        speech_quality={
            "flag": (
                llm_result.flag
                and not llm_result.error
            ),
            "fillers_detected":
                llm_result.fillers_detected,
            "colloquialisms_detected":
                llm_result.colloquialisms_detected,
            "orthoepic_errors_detected":
                llm_result.orthoepic_errors_detected,
            "examples":
                llm_result.examples,
            "confidence":
                round(llm_result.confidence, 3),
            "error":
                llm_result.error,
        },
        triggered_by=triggered,
    )
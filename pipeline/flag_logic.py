"""
Флаг «Логик» — качество речевого канала.

Компоненты:
  1. Технические метрики звука (audio_score из audio_quality модуля)
  2. LLM-анализ транскрипта: слова-паразиты, нелитературная речь
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field, asdict

from openai import OpenAI

logger = logging.getLogger(__name__)

# ─── Config ───────────────────────────────────────────────────────────────────

from config import DEEPSEEK_API_KEY, NARRATIVE_LLM_MODEL_CREATIVE as DEEPSEEK_MODEL

DEEPSEEK_BASE_URL     = "https://api.deepseek.com"
AUDIO_SCORE_THRESHOLD = 5.0

# ─── Prompt ───────────────────────────────────────────────────────────────────

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
class LogicLLMResult:
    flag: bool
    fillers_pattern: str
    fillers_detected: bool
    non_literary_pattern: str
    non_literary_detected: bool
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

    _empty = LogicLLMResult(
        flag=False,
        fillers_pattern="clean", fillers_detected=False,
        non_literary_pattern="clean", non_literary_detected=False,
    )

    if not DEEPSEEK_API_KEY:
        logger.warning("DEEPSEEK_API_KEY не задан — LLM-анализ пропущен")
        return LogicLLMResult(**{**_empty.__dict__, "error": "no_api_key"})

    client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)

    try:
        resp = client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            temperature=0.0,
            extra_body={"thinking": {"type": "enabled"}},
            messages=[
                {"role": "system", "content": LOGIC_LLM_PROMPT},
                {"role": "user",   "content": f"Транскрипт лекции:\n\n{transcript_text}"},
            ],
        )
        raw  = resp.choices[0].message.content.strip()
        data = json.loads(raw)

    except json.JSONDecodeError as e:
        logger.error(f"LLM вернул невалидный JSON: {e}")
        return LogicLLMResult(**{**_empty.__dict__, "error": f"json_parse_error: {e}"})
    except Exception as e:
        logger.error(f"Ошибка LLM-запроса: {e}")
        return LogicLLMResult(**{**_empty.__dict__, "error": str(e)})

    fillers_data      = data.get("fillers", {}) or {}
    non_literary_data = data.get("non_literary", {}) or {}

    fillers_pattern      = fillers_data.get("pattern", "clean")
    non_literary_pattern = non_literary_data.get("pattern", "clean")
    fillers_detected      = bool(fillers_data.get("detected", False))
    non_literary_detected = bool(non_literary_data.get("detected", False))

    confidence_map = {"clean": 0.85, "natural": 0.70, "excessive": 0.90}
    confidence = max(
        confidence_map.get(fillers_pattern, 0.75),
        confidence_map.get(non_literary_pattern, 0.75),
    )

    return LogicLLMResult(
        flag=fillers_detected or non_literary_detected,
        fillers_pattern=fillers_pattern,
        fillers_detected=fillers_detected,
        non_literary_pattern=non_literary_pattern,
        non_literary_detected=non_literary_detected,
        examples={
            "fillers":      fillers_data.get("examples", []),
            "non_literary": non_literary_data.get("examples", []),
        },
        confidence=float(confidence),
    )


# ─── Main entry point ─────────────────────────────────────────────────────────

def run(
    audio_score: float,
    transcript_text: str,
) -> LogicFlagResult:

    logger.info("── Флаг Логик: проверка качества звука ──")
    audio_flag = audio_score < AUDIO_SCORE_THRESHOLD
    triggered  = []
    if audio_flag:
        triggered.append(f"audio_score={audio_score:.2f} < {AUDIO_SCORE_THRESHOLD}")

    logger.info("── Флаг Логик: LLM-анализ речи ──")
    llm = _check_speech_quality(transcript_text)

    if llm.flag and not llm.error:
        if llm.fillers_detected:
            triggered.append(f"fillers ({llm.fillers_pattern})")
        if llm.non_literary_detected:
            triggered.append(f"non_literary ({llm.non_literary_pattern})")

    flag = audio_flag or (llm.flag and not llm.error)
    confidence = 0.95 if audio_flag else (llm.confidence if llm.flag else max(0.7, llm.confidence))

    return LogicFlagResult(
        flag=flag,
        confidence=round(confidence, 3),
        audio={
            "flag":      audio_flag,
            "score":     audio_score,
            "threshold": AUDIO_SCORE_THRESHOLD,
        },
        speech_quality={
            "flag":                  llm.flag,
            "fillers_pattern":       llm.fillers_pattern,
            "fillers_detected":      llm.fillers_detected,
            "non_literary_pattern":  llm.non_literary_pattern,
            "non_literary_detected": llm.non_literary_detected,
            "examples":              llm.examples,
            "confidence":            llm.confidence,
            "error":                 llm.error,
        },
        triggered_by=triggered,
    )
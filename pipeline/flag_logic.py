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

DNSMOS_THRESHOLD   = 2.5
SNR_THRESHOLD_DB   = 10.0
LUFS_THRESHOLD     = -30.0

# ─── LLM ──────────────────────────────────────────────────────────────────────
from config import DEEPSEEK_API_KEY, NARRATIVE_LLM_MODEL as DEEPSEEK_MODEL
 
DEEPSEEK_BASE_URL = "https://api.deepseek.com"

LOGIC_LLM_PROMPT = """\
Ты анализируешь транскрипт лекции на предмет качества речи лектора.

Оцени строго по трём критериям и ответь ТОЛЬКО валидным JSON без markdown:

{
  "fillers": {
    "detected": true/false,
    "examples": ["пример1", "пример2"]
  },
  "colloquialisms": {
    "detected": true/false,
    "examples": ["пример1"]
  },
  "orthoepic_errors": {
    "detected": true/false,
    "examples": ["пример1"]
  },
  "confidence": 0.0
}

Критерии:
- fillers: слова-паразиты («ну», «вот», «как бы», «э-э», «короче» и аналоги).
  Засчитывай только если встречаются часто, не единичные случаи.
- colloquialisms: просторечия и разговорные формы неуместные в лекции
  («ложить», «ихний», «звОнит», «вобщем» и т.п.)
- orthoepic_errors: грубые орфоэпические ошибки в записанной речи

НЕ засчитывай: обычный разговорный стиль подачи, технический жаргон,
единичные оговорки. Только систематические проблемы.

confidence — твоя уверенность в оценке от 0.0 до 1.0.
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


# ─── Audio quality check ──────────────────────────────────────────────────────

def _check_audio_quality(audio_quality: dict) -> LogicAudioResult:
    """Проверяем уже вычисленные метрики качества звука."""

    dnsmos   = audio_quality.get("dnsmos", {})
    lufs_val = audio_quality.get("lufs", {}).get("value", 0.0)
    snr_val  = audio_quality.get("snr", {}).get("db", 99.0)
    clip_lvl = audio_quality.get("clipping", {}).get("level", "none")

    ovrl = dnsmos.get("ovrl_mos", 5.0)
    sig  = dnsmos.get("sig_mos",  5.0)
    bak  = dnsmos.get("bak_mos",  5.0)

    triggered = []
    if ovrl < DNSMOS_THRESHOLD:
        triggered.append(f"ovrl_mos={ovrl:.2f} < {DNSMOS_THRESHOLD}")
    if sig < DNSMOS_THRESHOLD:
        triggered.append(f"sig_mos={sig:.2f} < {DNSMOS_THRESHOLD}")
    if bak < DNSMOS_THRESHOLD:
        triggered.append(f"bak_mos={bak:.2f} < {DNSMOS_THRESHOLD}")
    if snr_val < SNR_THRESHOLD_DB:
        triggered.append(f"snr={snr_val:.1f}dB < {SNR_THRESHOLD_DB}dB")
    if clip_lvl == "severe":
        triggered.append("clipping=severe")
    if lufs_val < LUFS_THRESHOLD:
        triggered.append(f"lufs={lufs_val:.1f} < {LUFS_THRESHOLD}")

    return LogicAudioResult(
        flag=len(triggered) > 0,
        ovrl_mos=ovrl,
        sig_mos=sig,
        bak_mos=bak,
        snr_db=snr_val,
        lufs=lufs_val,
        clipping=clip_lvl,
        triggered_by=triggered,
    )


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
    audio_quality: dict,
    transcript_text: str,
) -> LogicFlagResult:
    """
    Запускает флаг Логик.

    Args:
        audio_quality: результат модуля audio_quality (уже вычислен в pipeline)
        transcript_text: полный текст транскрипта

    Returns:
        LogicFlagResult
    """

    logger.info("── Флаг Логик: проверка качества звука ──")
    audio_result = _check_audio_quality(audio_quality)

    logger.info("── Флаг Логик: LLM-анализ речи ──")
    llm_result = _check_speech_quality(transcript_text)

    triggered = []
    if audio_result.flag:
        triggered.extend(audio_result.triggered_by)
    if llm_result.flag and not llm_result.error:
        if llm_result.fillers_detected:
            triggered.append("fillers")
        if llm_result.colloquialisms_detected:
            triggered.append("colloquialisms")
        if llm_result.orthoepic_errors_detected:
            triggered.append("orthoepic_errors")

    flag = audio_result.flag or (llm_result.flag and not llm_result.error)

    # Уверенность: если аудио сработало — уверенность высокая (метрики детерминированы)
    if audio_result.flag:
        confidence = 0.95
    elif llm_result.flag:
        confidence = llm_result.confidence
    else:
        confidence = max(0.7, llm_result.confidence)

    return LogicFlagResult(
        flag=flag,
        confidence=confidence,
        audio={
            "flag":         audio_result.flag,
            "ovrl_mos":     audio_result.ovrl_mos,
            "sig_mos":      audio_result.sig_mos,
            "bak_mos":      audio_result.bak_mos,
            "snr_db":       audio_result.snr_db,
            "lufs":         audio_result.lufs,
            "clipping":     audio_result.clipping,
            "triggered_by": audio_result.triggered_by,
        },
        speech_quality={
            "flag":                      llm_result.flag,
            "fillers_detected":          llm_result.fillers_detected,
            "colloquialisms_detected":   llm_result.colloquialisms_detected,
            "orthoepic_errors_detected": llm_result.orthoepic_errors_detected,
            "examples":                  llm_result.examples,
            "confidence":                llm_result.confidence,
            "error":                     llm_result.error,
        },
        triggered_by=triggered,
    )

"""
Флаг «Интуит» — смысловая динамика и подача материала.

Компоненты:
  1. LLM-анализ транскрипта: вброс в начале, метафоры, перефразирование,
     академизм, механические отговорки
  2. OCR слайдов/доски: дублирование текста экрана в речи (YOLO-E + EasyOCR + BLEU)
"""

from __future__ import annotations

import json
import logging
import os
import base64
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
from openai import OpenAI

logger = logging.getLogger(__name__)

# ─── LLM ──────────────────────────────────────────────────────────────────────
from config import DEEPSEEK_API_KEY, NARRATIVE_LLM_MODEL_CREATIVE as DEEPSEEK_MODEL
 
DEEPSEEK_BASE_URL = "https://api.deepseek.com"

INTUITIVE_LLM_PROMPT = """\
Ты оцениваешь смысловую динамику лекции с точки зрения интуитивного типа.

КОНТЕКСТ. Интуит воспринимает лекцию через движение и плотность смыслов:
важно чтобы мысль развивалась, появлялись новые ракурсы и связи, а лектор
говорил своим голосом, а не пересказывал учебник. Скучно становится когда
смысл топчется на месте — лектор переливает воду, дословно дублирует
сказанное, читает с экрана.

Флаг должен срабатывать только когда лекция системно не удовлетворяет
этой потребности, а не из-за отдельных моментов. Уместные повторы для
закрепления, паузы, переходы между темами — часть нормальной подачи, не
вода. Лекция может быть фрагментом курса или стрима — отсутствие
зрелищного начала не является дефектом само по себе.

Ответь ТОЛЬКО валидным JSON без markdown:

{
  "semantic_dynamics": {
    "pattern": "strong | present | absent",
    "detected": true/false,
    "examples": ["цитата из транскрипта", "..."]
  },
  "imagery": {
    "pattern": "strong | present | absent",
    "detected": true/false,
    "examples": ["..."]
  },
  "own_voice": {
    "pattern": "strong | present | absent",
    "detected": true/false,
    "examples": ["..."]
  }
}

Что оцениваем:

- semantic_dynamics: движется ли мысль вперёд. Каждый следующий фрагмент
  лекции добавляет новое — поворот, обобщение, неожиданный ракурс, выход
  на следующий уровень — или лектор переливает из пустого в порожнее,
  дублирует сказанное другими словами, зачитывает то что и так на экране.
  Это центральная ось — именно она про «уже понял, а лектор продолжает».

- imagery: использует ли лектор метафоры, аналогии, переносы понятий из
  одной области в другую, образные сравнения. Не считается одно техническое
  определение или вычисление — речь о том, помогает ли лектор увидеть
  предмет с разных сторон через образы.

- own_voice: слышен ли в лекции сам лектор — его взгляд, интерпретация,
  оценка, личный пример, неожиданный заход на тему. Или содержание можно
  слово в слово прочитать в любом учебнике по предмету.

Шкала pattern (общая для всех трёх):

- strong — качество явно выражено и держится по всей лекции, лектор
  работает на этой оси сознательно.

- present — качество есть в нормальном объёме, не выделяется как фишка
  подачи, но и не отсутствует. Обычная живая лекция.

- absent — качество практически отсутствует, и это не обусловлено
  форматом (формальный вывод теоремы не обязан быть образным, пошаговый
  разбор кода — иметь авторский голос). Для semantic_dynamics absent
  ставится только если виден устойчивый паттерн топтания на всей
  лекции, а не отдельные повторы.

ПРАВИЛО: detected = true только при pattern = "absent".
detected означает обнаруженную проблему интуита, а не качество.

В examples — короткие цитаты из транскрипта (1–3 штуки) подтверждающие
вердикт. Для absent — фрагменты где видно отсутствие или топтание;
для strong и present — фрагменты подтверждающие наличие. Пустой список
допустим, если короткая цитата вне контекста ничего не показывает.

НЕ засчитывай как absent:
- уместные повторы и педагогические приёмы закрепления (лектор
  сознательно возвращается к ключевой мысли);
- паузы, переходы, организационные ремарки между блоками;
- формат, который сам по себе не требует образности или авторского
  голоса (вывод формулы, пошаговый разбор кода);
- отсутствие зрелищного начала — лекция может быть фрагментом курса
  или стрима;
- собственные оценочные ремарки типа «как я уже говорил» — часть устной
  речи, а не маркер механической подачи;
- технические повторы лектора (когда сам что-то проговаривает повторно
  без вопроса из зала) — это не дублирование, это часть подачи.
"""

# ─── OCR thresholds ───────────────────────────────────────────────────────────

BLEU_THRESHOLD_NORMAL       = 0.55
BLEU_THRESHOLD_INSTRUMENTAL = 0.35
SPECIAL_CHAR_DENSITY_THRESH = 0.20
MIN_WORDS_FOR_COMPARE       = 5

# Минимальное число проблемных критериев LLM чтобы сработал флаг
LLM_ISSUES_THRESHOLD = 2


# ─── Data structures ──────────────────────────────────────────────────────────

@dataclass
class IntuitiveLLMResult:
    issues_count: int
    hook_in_opening: bool
    metaphors_analogies: bool
    reformulation: bool
    synthesis_not_summary: bool
    own_perspective: bool
    mechanical_dismissal: bool
    details: dict = field(default_factory=dict)
    confidence: float = 0.0
    error: str | None = None


@dataclass
class IntuitiveOCRResult:
    checked: bool
    flag: bool
    segments_checked: int
    segments_triggered: int
    is_instrumental: bool
    error: str | None = None


@dataclass
class IntuitiveFlagResult:
    flag: bool
    confidence: float
    llm: dict
    ocr: dict
    triggered_by: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


# ─── LLM analysis ─────────────────────────────────────────────────────────────

def _run_llm(transcript_text: str) -> IntuitiveLLMResult:
    """LLM-анализ транскрипта на смысловую динамику."""

    if not DEEPSEEK_API_KEY:
        logger.warning("DEEPSEEK_API_KEY не задан — LLM-анализ пропущен")
        return IntuitiveLLMResult(
            issues_count=0, hook_in_opening=True, metaphors_analogies=True,
            reformulation=True, synthesis_not_summary=True, own_perspective=True,
            mechanical_dismissal=False, error="no_api_key",
        )

    client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)
    text_sample = transcript_text

    try:
        resp = client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            temperature=0.0,
            messages=[
                {"role": "system", "content": INTUITIVE_LLM_PROMPT},
                {"role": "user",   "content": f"Транскрипт лекции:\n\n{text_sample}"},
            ],
        )
        raw = resp.choices[0].message.content.strip()
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        logger.error(f"LLM вернул невалидный JSON: {e}")
        return IntuitiveLLMResult(
            issues_count=0, hook_in_opening=True, metaphors_analogies=True,
            reformulation=True, synthesis_not_summary=True, own_perspective=True,
            mechanical_dismissal=False, error=f"json_parse_error: {e}",
        )
    except Exception as e:
        logger.error(f"Ошибка LLM-запроса: {e}")
        return IntuitiveLLMResult(
            issues_count=0, hook_in_opening=True, metaphors_analogies=True,
            reformulation=True, synthesis_not_summary=True, own_perspective=True,
            mechanical_dismissal=False, error=str(e),
        )

    hook       = data.get("hook_in_opening",      {}).get("detected", True)
    metaphors  = data.get("metaphors_analogies",  {}).get("detected", True)
    reformat   = data.get("reformulation",        {}).get("detected", True)
    synthesis  = data.get("synthesis_not_summary",{}).get("detected", True)
    own        = data.get("own_perspective",      {}).get("detected", True)
    mechanical = data.get("mechanical_dismissal", {}).get("detected", False)

    # Считаем проблемы: плохо когда хороших качеств нет или есть отговорки
    issues = sum([
        not hook,
        not metaphors,
        not reformat,
        not synthesis,
        not own,
        mechanical,
    ])

    return IntuitiveLLMResult(
        issues_count=issues,
        hook_in_opening=hook,
        metaphors_analogies=metaphors,
        reformulation=reformat,
        synthesis_not_summary=synthesis,
        own_perspective=own,
        mechanical_dismissal=mechanical,
        details=data,
        confidence=float(data.get("confidence", 0.5)),
    )


# ─── OCR slide comparison ─────────────────────────────────────────────────────

def _is_instrumental(ocr_text: str) -> bool:
    """Инструментальное видео — код, формулы, нотация."""
    words = ocr_text.split()
    if len(words) < MIN_WORDS_FOR_COMPARE:
        return True
    special = sum(1 for c in ocr_text if c in "={}()<>[]+-*/\\^|#@$%&;:")
    density = special / max(len(ocr_text), 1)
    return density > SPECIAL_CHAR_DENSITY_THRESH


def _bleu_bigram(reference_text: str, hypothesis_text: str) -> float:
    """Простой BLEU на биграмах без внешних зависимостей."""
    try:
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        ref_tokens = reference_text.lower().split()
        hyp_tokens = hypothesis_text.lower().split()
        if not ref_tokens or not hyp_tokens:
            return 0.0
        score = sentence_bleu(
            [ref_tokens], hyp_tokens,
            weights=(0.5, 0.5, 0, 0),
            smoothing_function=SmoothingFunction().method1,
        )
        return float(score)
    except ImportError:
        # Fallback: простой n-gram overlap
        ref_words = set(reference_text.lower().split())
        hyp_words = set(hypothesis_text.lower().split())
        if not ref_words:
            return 0.0
        return len(ref_words & hyp_words) / len(ref_words)


def _run_ocr(frame_paths: list[str], transcript_segments: list[dict]) -> IntuitiveOCRResult:
    """
    Проверяем дублирование: лектор читает то что на слайде.

    Args:
        frame_paths: пути к кадрам видео (уже извлечённым)
        transcript_segments: [{"text": str, "start": float, "end": float}, ...]
    """

    if not frame_paths:
        return IntuitiveOCRResult(checked=False, flag=False,
                                  segments_checked=0, segments_triggered=0,
                                  is_instrumental=False, error="no_frames")

    try:
        import easyocr
        import cv2
    except ImportError:
        return IntuitiveOCRResult(checked=False, flag=False,
                                  segments_checked=0, segments_triggered=0,
                                  is_instrumental=False, error="easyocr_not_installed")

    # YOLO-E детекция слайда — если нет, используем весь кадр
    # (YOLO-E интеграция предполагается отдельно, здесь fallback на весь кадр)
    reader = easyocr.Reader(["ru", "en"], gpu=True, verbose=False)

    triggered = 0
    checked   = 0
    is_instr  = False

    for frame_path in frame_paths:
        frame = cv2.imread(frame_path)
        if frame is None:
            continue

        # OCR
        results = reader.readtext(frame_path, detail=0)
        ocr_text = " ".join(results).strip()

        if len(ocr_text.split()) < MIN_WORDS_FOR_COMPARE:
            continue

        is_instr = _is_instrumental(ocr_text)
        threshold = BLEU_THRESHOLD_INSTRUMENTAL if is_instr else BLEU_THRESHOLD_NORMAL

        # Находим соответствующий сегмент транскрипта по времени кадра
        # (упрощение: берём весь транскрипт за ближайший временной отрезок)
        # В реальной интеграции frame_path содержит timestamp в имени
        transcript_chunk = " ".join([s["text"] for s in transcript_segments])[:500]

        score = _bleu_bigram(ocr_text, transcript_chunk)
        checked += 1

        if score > threshold:
            triggered += 1
            logger.debug(f"OCR дублирование: BLEU={score:.2f} (порог={threshold})")

    flag = (checked > 0) and (triggered / max(checked, 1) > 0.4)

    return IntuitiveOCRResult(
        checked=True,
        flag=flag,
        segments_checked=checked,
        segments_triggered=triggered,
        is_instrumental=is_instr,
    )


# ─── Main entry point ─────────────────────────────────────────────────────────

def run(
    transcript_text: str,
    transcript_segments: list[dict] | None = None,
    frame_paths: list[str] | None = None,
) -> IntuitiveFlagResult:
    """
    Запускает флаг Интуит.

    Args:
        transcript_text: полный текст транскрипта
        transcript_segments: [{"text": str, "start": float, "end": float}]
        frame_paths: пути к кадрам для OCR (опционально)

    Returns:
        IntuitiveFlagResult
    """

    logger.info("── Флаг Интуит: LLM-анализ смысловой динамики ──")
    llm = _run_llm(transcript_text)

    ocr_result = IntuitiveOCRResult(
        checked=False, flag=False, segments_checked=0,
        segments_triggered=0, is_instrumental=False,
    )
    if frame_paths and transcript_segments:
        logger.info("── Флаг Интуит: OCR проверка слайдов ──")
        ocr_result = _run_ocr(frame_paths, transcript_segments)

    triggered = []
    if not llm.error:
        if not llm.hook_in_opening:       triggered.append("no_hook_in_opening")
        if not llm.metaphors_analogies:   triggered.append("no_metaphors")
        if not llm.reformulation:         triggered.append("no_reformulation")
        if not llm.synthesis_not_summary: triggered.append("no_synthesis")
        if not llm.own_perspective:       triggered.append("no_own_perspective")
        if llm.mechanical_dismissal:      triggered.append("mechanical_dismissal")
    if ocr_result.flag:
        triggered.append("slide_reading_detected")

    llm_flag = (not llm.error) and (llm.issues_count >= LLM_ISSUES_THRESHOLD)
    flag = llm_flag or ocr_result.flag

    confidence = llm.confidence if not llm.error else 0.0
    if ocr_result.flag:
        confidence = max(confidence, 0.8)

    return IntuitiveFlagResult(
        flag=flag,
        confidence=round(confidence, 3),
        llm={
            "flag":                  llm_flag,
            "issues_count":          llm.issues_count,
            "hook_in_opening":       llm.hook_in_opening,
            "metaphors_analogies":   llm.metaphors_analogies,
            "reformulation":         llm.reformulation,
            "synthesis_not_summary": llm.synthesis_not_summary,
            "own_perspective":       llm.own_perspective,
            "mechanical_dismissal":  llm.mechanical_dismissal,
            "details":               llm.details,
            "confidence":            llm.confidence,
            "error":                 llm.error,
        },
        ocr={
            "checked":            ocr_result.checked,
            "flag":               ocr_result.flag,
            "segments_checked":   ocr_result.segments_checked,
            "segments_triggered": ocr_result.segments_triggered,
            "is_instrumental":    ocr_result.is_instrumental,
            "error":              ocr_result.error,
        },
        triggered_by=triggered,
    )

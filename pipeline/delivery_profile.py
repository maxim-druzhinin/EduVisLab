"""
Модуль профиля подачи: академичность и инструментальность.

Один LLM-вызов с мышлением на полном транскрипте.
Prompt и настройки проверены — не менять.
"""

import json
import logging
from dataclasses import dataclass
from typing import Optional

from openai import OpenAI

from config import DEEPSEEK_API_KEY, DELIVERY_PROFILE_MODEL

logger = logging.getLogger(__name__)


# ─── Lazy singleton ───────────────────────────────────────────────────────────

_llm_client: Optional[OpenAI] = None


def _get_llm_client() -> OpenAI:
    global _llm_client
    if _llm_client is None:
        _llm_client = OpenAI(
            api_key=DEEPSEEK_API_KEY,
            base_url="https://api.deepseek.com",
        )
    return _llm_client


# ─── Data classes ─────────────────────────────────────────────────────────────

@dataclass
class DeliveryProfileResult:
    # Уровни 1–5
    academic_level:       int
    instrumental_level:   int

    # Скоры 0.0–1.0
    academic_score:       float
    instrumental_score:   float

    # Производные
    bias:       float          # academic_score - instrumental_score
    intensity:  float          # (academic_score + instrumental_score) / 2
    dominant:   str            # "academic" | "instrumental" | "mixed" | "weak"

    # Объяснения
    evidence_academic:    list[str]
    evidence_instrumental: list[str]
    rationale:            str


# ─── Helpers ──────────────────────────────────────────────────────────────────

_LEVEL_SCORE_RANGES = {
    1: (0.00, 0.20),
    2: (0.20, 0.40),
    3: (0.40, 0.65),
    4: (0.65, 0.85),
    5: (0.85, 1.00),
}


def _fmt_duration(seconds: float) -> str:
    seconds = int(seconds or 0)
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def _clamp01(x, default: float = 0.0) -> float:
    try:
        x = float(x)
    except Exception:
        x = default
    return max(0.0, min(1.0, x))


def _clamp_level(x, default: int = 1) -> int:
    try:
        x = int(round(float(x)))
    except Exception:
        x = default
    return max(1, min(5, x))


def _score_from_level(level: int) -> float:
    lo, hi = _LEVEL_SCORE_RANGES[level]
    return round((lo + hi) / 2, 3)


def _align_score_with_level(score: float, level: int) -> float:
    score = _clamp01(score)
    lo, hi = _LEVEL_SCORE_RANGES[level]
    soft_lo = max(0.0, lo - 0.05)
    soft_hi = min(1.0, hi + 0.05)
    return round(min(max(score, soft_lo), soft_hi), 3)


def _infer_dominant(academic: float, instrumental: float) -> str:
    if academic < 0.3 and instrumental < 0.3:
        return "weak"
    if abs(academic - instrumental) <= 0.15 and max(academic, instrumental) >= 0.45:
        return "mixed"
    return "academic" if academic > instrumental else "instrumental"


# ─── Prompt ───────────────────────────────────────────────────────────────────

_SYSTEM = (
    "Ты эксперт по анализу образовательных видео и педагогического стиля подачи. "
    "Оценивай не качество видео и не сложность темы, а глобальную педагогическую цель занятия. "
    "Верни результат строго в JSON. Без текста вне JSON. Без markdown-блоков."
)


def _build_prompt(metadata: dict, transcript_text: str, user_params: dict) -> str:
    level    = user_params.get("level", "не указан")
    immersion = user_params.get("immersion", "не указана")
    view_goal = user_params.get("view_goal", "не указана")

    return f"""=== МЕТАДАННЫЕ ===
Название видео: {metadata['title']}
Канал: {metadata.get('channel', 'не указан')}
Длительность: {_fmt_duration(metadata.get('duration', 0))}

Параметры пользователя:
- Уровень: {level}
- Погружённость: {immersion}
- Цель просмотра: {view_goal}

Метаданные используй только как контекст. Оценки ставь по фактической подаче в транскрипте.

=== ЧТО НУЖНО ИЗМЕРИТЬ ===

Оцени не качество видео, не сложность темы, не престиж источника и не формальность языка, а глобальную педагогическую цель занятия.

Есть две НЕЗАВИСИМЫЕ оси:

1. АКАДЕМИЧНОСТЬ

Академичность — насколько видео стремится построить у зрителя теоретическое понимание предмета: язык, понятия, модели, основания, связи между объектами, обобщения, доказательства, внутреннюю логику темы.

Академический полюс — это "понять почему":
- почему объект определяется именно так;
- почему утверждение верно;
- почему метод работает;
- как устроена внутренняя логика предмета;
- как рассуждать строго.

2. ИНСТРУМЕНТАЛЬНОСТЬ

Инструментальность — насколько видео стремится научить зрителя действовать в предметной области: решать задачи, применять метод, выполнять процедуру, разбирать кейсы, делать вычисления, писать код, готовиться к контрольной, экзамену, проекту или практическому использованию.

Инструментальный полюс — это "уметь как":
- как решить задачу;
- как применить формулу или метод;
- как выполнить процедуру;
- как посчитать, построить, настроить, использовать;
- как действовать в похожем случае.

Оси независимы. Видео может быть одновременно академичным и инструментальным.

=== ШКАЛА АКАДЕМИЧНОСТИ: academic_level 1–5 ===

1 = разговорное или обзорное объяснение "на пальцах": почти нет строгих терминов, формального аппарата, теоретической рамки или обоснований.

2 = концептуальное объяснение: термины и идеи есть, но без строгой системы, формальных определений, доказательств или глубокого вывода.

3 = умеренно академическая подача: определения, формулы, понятия или модели вводятся достаточно аккуратно; есть объяснение связей, но теория не строится как строгая система.

4 = строгая академическая подача: есть определения, утверждения, обоснования, доказательства или выводы; цель — понять, почему утверждения верны, и научиться рассуждать строго.

5 = фундаментальная теоретическая подача: предмет строится как система; есть аксиоматика, формальные доказательства, глубокие обобщения, строгая структура понятий; прикладная тренировка минимальна или вторична.

=== ШКАЛА ИНСТРУМЕНТАЛЬНОСТИ: instrumental_level 1–5 ===

1 = почти нет применения: видео в основном объясняет, обсуждает или мотивирует, но не учит действовать, решать, считать или применять.

2 = есть отдельные примеры или намёки на применение, но без полноценного разбора метода, задачи, процедуры или результата.

3 = умеренная применимость: есть разбор метода, примеры, ход рассуждения, частичная демонстрация применения или пошаговое объяснение, которое можно повторить.

4 = практико-ориентированная подача: значимая часть видео посвящена задачам, кейсам, вычислениям, алгоритмам, типовым ситуациям, ошибкам или применению метода; зритель понимает, как применять материал.

5 = tutorial / practicum / problem-solving session: главная цель — повторить действия и получить результат; много конкретных шагов, процедур, решений, вычислений, кода, настроек или практических действий.

=== КАЛИБРОВКА ===

Эталон высокой академичности — фундаментальная лекция, где строится строгая структура предмета: определения, доказательства, связи, основания.

Эталон высокой инструментальности — практикум, tutorial или семинар, где зрителя учат действовать: решать, считать, применять, повторять процедуру и получать результат.

Пошаговое теоретическое объяснение может иметь высокую академичность и среднюю инструментальность.

Семинар высокого уровня с теорией и задачами может быть высоким по обеим осям.

=== SCORE ВНУТРИ LEVEL ===

Сначала выбери уровень 1–5 по каждой оси, затем поставь score 0.0–1.0 как тонкую позицию внутри выбранного уровня.

Ориентировочное соответствие:
- level 1 → score обычно 0.00–0.20
- level 2 → score обычно 0.20–0.40
- level 3 → score обычно 0.40–0.65
- level 4 → score обычно 0.65–0.85
- level 5 → score обычно 0.85–1.00

Score может быть немного ниже или выше диапазона только если видео находится на границе уровней, но level и score не должны противоречить друг другу.

=== ВАЖНО ===

- Академичность и инструментальность — не шкала качества.
- Академичность не значит "лучше", инструментальность не значит "хуже".
- Не завышай обе оценки только потому, что видео образовательное.
- Не оценивай по названию вуза, канала, автора или сложности темы.
- Локальные признаки вроде определений, формул, примеров, слов "берём/получаем/подставляем" используй только как evidence. Они не должны автоматически определять оценку.
- Один пример не делает видео инструментальным на 5.
- Одна формула не делает видео академичным на 5.
- Подробное доказательство или логический разбор может давать среднюю инструментальность, если зритель учится воспроизводить ход рассуждения.
- Оцени всё видео в целом, а не самый яркий фрагмент.
- Если видео в основном состоит из вступления, истории, мотивации или рекомендаций, обе оценки должны быть ниже.

=== ВЕРНИ СТРОГО JSON ===

{{
  "academic_level": <целое число 1..5>,
  "academic_score": <число 0.0..1.0>,
  "instrumental_level": <целое число 1..5>,
  "instrumental_score": <число 0.0..1.0>,
  "evidence_academic": [
    "<короткая цитата или точное описание признака из транскрипта>"
  ],
  "evidence_instrumental": [
    "<короткая цитата или точное описание признака из транскрипта>"
  ],
  "rationale": "<2-4 предложения: почему выбраны такие уровни и оценки>"
}}

=== ТРАНСКРИПТ ===
{transcript_text}"""


# ─── LLM call ─────────────────────────────────────────────────────────────────

def _llm_call(system: str, user: str) -> dict:
    client = _get_llm_client()

    response = client.chat.completions.create(
        model=DELIVERY_PROFILE_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        temperature=0.0,
        response_format={"type": "json_object"},
        extra_body={"thinking": {"type": "disabled"}},
    )

    content = response.choices[0].message.content
    if not content or not content.strip():
        raise RuntimeError("LLM вернул пустой ответ")

    return json.loads(content)


# ─── Assembly ─────────────────────────────────────────────────────────────────

def _assemble(raw: dict) -> DeliveryProfileResult:
    academic_level    = _clamp_level(raw.get("academic_level"))
    instrumental_level = _clamp_level(raw.get("instrumental_level"))

    academic_score = (
        _align_score_with_level(raw["academic_score"], academic_level)
        if raw.get("academic_score") is not None
        else _score_from_level(academic_level)
    )
    instrumental_score = (
        _align_score_with_level(raw["instrumental_score"], instrumental_level)
        if raw.get("instrumental_score") is not None
        else _score_from_level(instrumental_level)
    )

    return DeliveryProfileResult(
        academic_level=academic_level,
        instrumental_level=instrumental_level,
        academic_score=academic_score,
        instrumental_score=instrumental_score,
        bias=round(academic_score - instrumental_score, 3),
        intensity=round((academic_score + instrumental_score) / 2, 3),
        dominant=_infer_dominant(academic_score, instrumental_score),
        evidence_academic=raw.get("evidence_academic") or [],
        evidence_instrumental=raw.get("evidence_instrumental") or [],
        rationale=raw.get("rationale") or "",
    )


# ─── Main entry point ─────────────────────────────────────────────────────────

def run(
    transcript_text: str,
    metadata: dict,
    user_params: dict,
) -> DeliveryProfileResult:

    if not transcript_text.strip():
        raise ValueError("Транскрипт пустой")

    logger.info(
        f"Delivery profile: {len(transcript_text.split())} слов, "
        f"модель={DELIVERY_PROFILE_MODEL}"
    )

    prompt = _build_prompt(metadata, transcript_text, user_params)
    raw    = _llm_call(_SYSTEM, prompt)

    return _assemble(raw)

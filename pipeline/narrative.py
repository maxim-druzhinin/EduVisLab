"""
Модуль сегментации и адаптивного нарратива.

Шаги:
  1. light_merge      — whisper сегменты → ~50-секундные чанки с block_id
  2. LLM #1 (T=0.0)  — segment extraction: LLM сама определяет границы тем
  3. LLM #2 (T=0.0)  — educational analysis: Bloom's, пресреквизиты и т.д.
  4. LLM #3 (T=0.7)  — adaptive narrative: финальный текст для пользователя
"""

import json
import logging
from dataclasses import dataclass
from typing import Optional

from openai import OpenAI

from config import DEEPSEEK_API_KEY, NARRATIVE_LLM_MODEL, NARRATIVE_LIGHT_MERGE_SEC, NARRATIVE_LLM_MODEL_CREATIVE, SEGMENT_MODEL_BY_GOAL

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
class NarrativeSegment:
    title: str
    description: str
    start: float          # таймкод в секундах, точный (из whisper)


@dataclass
class BloomsLabel:
    segment_title: str
    level: str            # remember / understand / apply / analyze / evaluate / create
    evidence: str


@dataclass
class Prerequisite:
    concept: str
    confidence: str       # high / medium / low


@dataclass
class NarrativeResult:
    # Тематические сегменты
    segments: list[NarrativeSegment]

    # Educational analysis
    blooms_per_segment: list[BloomsLabel]
    blooms_dominant: str
    prerequisites: list[Prerequisite]
    learning_path_type: str
    learning_path_reasoning: str
    topics_covered: list[str]
    topics_gaps: list[str]
    info_density: str
    unique_approach: Optional[str]
    title_match_verdict: str
    title_match_explanation: str

    # Финальный текст
    narrative: str

    # Техническое
    n_chunks: int
    n_segments: int


# ─── Step 1: Light merge ──────────────────────────────────────────────────────

def light_merge(
    segments: list[dict],
    target_duration: float = NARRATIVE_LIGHT_MERGE_SEC,
) -> list[dict]:
    """
    Склеивает whisper-сегменты в блоки ~target_duration секунд.
    Присваивает каждому блоку block_id для маппинга таймкодов.

    Входной формат:  [{"text": str, "start": float, "end": float}, ...]
    Выходной формат: [{"block_id": str, "start": float, "text": str}, ...]
    """
    if not segments:
        return []

    chunks: list[dict] = []
    current_text: list[str] = []
    current_start = segments[0]["start"]

    for i, seg in enumerate(segments):
        current_text.append(seg["text"].strip())
        chunk_len = seg["end"] - current_start

        if chunk_len >= target_duration or i == len(segments) - 1:
            chunks.append({
                "block_id": f"B{len(chunks) + 1:02d}",
                "start":    current_start,
                "text":     " ".join(current_text),
            })
            current_text = []
            if i + 1 < len(segments):
                current_start = segments[i + 1]["start"]

    logger.info(f"light_merge: {len(segments)} сегментов → {len(chunks)} чанков")
    return chunks


# ─── Formatting helpers ───────────────────────────────────────────────────────

def _fmt_timestamp(seconds: float) -> str:
    m, s = int(seconds) // 60, int(seconds) % 60
    return f"{m}:{s:02d}"


def _fmt_duration(seconds: float) -> str:
    h = int(seconds) // 3600
    m = (int(seconds) % 3600) // 60
    s = int(seconds) % 60
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def _fmt_chunks_for_llm(chunks: list[dict]) -> str:
    lines = []
    for c in chunks:
        lines.append(f"[{c['block_id']} · {_fmt_timestamp(c['start'])}]\n{c['text']}")
    return "\n\n".join(lines)


# ─── LLM helper ───────────────────────────────────────────────────────────────

def _llm_call(
    system: str,
    user: str,
    temperature: float,
    model: str = NARRATIVE_LLM_MODEL,
    json_mode: bool = False,
    retries: int = 3,
) -> str:
    client = _get_llm_client()

    kwargs: dict = {
        "model":       model,
        "messages":    [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        "temperature": temperature,
        "extra_body":  {"thinking": {"type": "disabled"}},
    }
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    for attempt in range(1, retries + 1):
        response = client.chat.completions.create(**kwargs)
        content = response.choices[0].message.content

        if content and content.strip():
            return content

        logger.warning(
            f"Попытка {attempt}/{retries}: пустой ответ "
            f"(finish_reason={response.choices[0].finish_reason})"
        )

    raise RuntimeError("LLM вернул пустой ответ после всех попыток")


# ─── LLM #1: Segment Extraction ───────────────────────────────────────────────

_SEGMENT_SYSTEM = (
    "Ты структурируешь транскрипт лекции на тематические части. "
    "Верни результат строго в JSON. Без текста вне JSON. Без markdown-блоков."
)

_SEGMENT_INSTRUCTIONS = {
    "Составить общее представление": (
        "Выделяй только крупные смысловые части. "
        "Объединяй близкие подтемы, если они служат одной общей идее. "
        "Не дроби определения, примеры и пояснения, если они относятся к одному блоку понимания. "
        "Описание короткое: что в целом происходит в этой части."
    ),

    "Закрыть точечные вопросы": (
        "Дроби подробнее: отделяй новые определения, важные примеры, доказательства, переходы к частным вопросам. "
        "Пользователь должен по названию и описанию понять, где искать конкретный ответ. "
        "Не объединяй разные самостоятельные вопросы только потому, что они рядом по времени."
    ),

    "Последовательно изучить тему": (
        "Выделяй учебные главы среднего размера. "
        "Отделяй подготовку, введение понятий, основные определения, примеры, доказательства и выводы. "
        "Не дроби до отдельных реплик, но и не склеивай этапы, которые требуют разного понимания."
    ),
}


def _build_segment_prompt(
    metadata: dict,
    chunks: list[dict],
    user_params: dict,
) -> str:
    chapters_str = str(metadata.get("chapters")) if metadata.get("chapters") else "не указаны"
    instruction = _SEGMENT_INSTRUCTIONS.get(user_params.get("view_goal", ""), "")

    return f"""=== МЕТАДАННЫЕ ===
Название: {metadata['title']}
Длительность: {_fmt_duration(metadata.get('duration', 0))}
Главы от автора: {chapters_str}

=== ТРАНСКРИПТ ===
{_fmt_chunks_for_llm(chunks)}

=== ЗАДАЧА ===
{instruction}

Определи тематические части самостоятельно по тексту — ищи явные переходы
("итак", "теперь", "перейдём", смена темы по содержанию).

Гранулярность выбирай адаптивно: учитывай длительность видео, плотность объяснения и количество реальных переходов.
Не стремись к фиксированному числу частей.
Если тема развивается плавно, объединяй; если начинается новый учебный шаг, отделяй.

Используй block_id строго из списка выше.

{{
  "segments": [
    {{
      "block_id_start": "<block_id первого блока этой темы>",
      "title":          "<короткое название темы, 3-6 слов>",
      "description":    "<описание согласно инструкции выше>"
    }}
  ]
}}"""


def _run_segment_extraction(
    metadata: dict,
    chunks: list[dict],
    user_params: dict,
) -> list[dict]:
    cfg = SEGMENT_MODEL_BY_GOAL.get(
        user_params.get("view_goal", ""),
        {"model": NARRATIVE_LLM_MODEL, "thinking": False},
    )
    logger.info(f"LLM #1: segment extraction ({cfg['model']}, thinking={cfg['thinking']})...")
    raw = _llm_call(
        system=_SEGMENT_SYSTEM,
        user=_build_segment_prompt(metadata, chunks, user_params),
        temperature=0.0,
        json_mode=True,
        model=cfg["model"],
        thinking=cfg["thinking"],
        #тут нормально справляется flash, с ризонингом она сходит с ума, pro без ризонинга наверное чуть лучше, она больше переформулирует, pro с ризонингом хорошо собирает "общее представление", прям оч крупными мазками, но время инференса просто космическое на длинной лекции
    )
    return json.loads(raw)["segments"]

# Возожно в будущем сделать вот так: 
# SEGMENT_MODEL_BY_GOAL = {
#     "Закрыть точечные вопросы":      {"model": "deepseek-v4-flash", "thinking": False},
#     "Последовательно изучить тему":  {"model": "deepseek-v4-flash", "thinking": False},
#     "Составить общее представление":  {"model": "deepseek-v4-pro",   "thinking": True},
# }

# ─── LLM #2: Educational Analysis ────────────────────────────────────────────

_ANALYSIS_SYSTEM = (
    "Ты анализируешь образовательное видео с педагогической точки зрения. "
    "Верни результат строго в JSON. Без текста вне JSON. Без markdown-блоков."
)


def _build_analysis_prompt(
    metadata: dict,
    segments: list[dict],
    chunks: list[dict],
    user_params: dict,
) -> str:
    segments_str = "\n".join(
        f"- {s['title']}: {s.get('description', '')}" for s in segments
    )

    return f"""=== КОНТЕКСТ ПОЛЬЗОВАТЕЛЯ ===
Уровень: {user_params.get('level')}
Погружённость: {user_params.get('immersion')}
Цель просмотра: {user_params.get('view_goal')}

Учитывай этот контекст при анализе — в частности при определении пресреквизитов.

=== МЕТАДАННЫЕ ===
Название: {metadata['title']}
Описание: {(metadata.get('description') or '')[:800]}
Канал: {metadata.get('channel') or 'не указан'}
Плейлист: {metadata.get('playlist') or 'не указан'}, позиция: {metadata.get('playlist_index') or 'не указана'}

=== ТЕМЫ ВИДЕО ===
{segments_str}

=== ПОЛНЫЙ ТРАНСКРИПТ ===
{_fmt_chunks_for_llm(chunks)}

=== ЗАДАЧА ===
{{
  "blooms_per_segment": [
    {{
      "title":    "<название темы из списка выше>",
      "level":    "<remember|understand|apply|analyze|evaluate|create>",
      "evidence": "<1 пример глагола или фразы из транскрипта>"
    }}
  ],

  "prerequisites": [
    {{"concept": "<понятие>", "confidence": "<high|medium|low>"}}
  ],

  "learning_path": {{
    "type":      "<intro|part_of_course|standalone|advanced>",
    "reasoning": "<одно предложение>"
  }},

  "topic_coverage": {{
    "covered": ["<темы которые реально разобраны>"],
    "gaps":    ["<темы из названия которые не раскрыты, или пустой список>"]
  }},

  "info_density":    "<low|medium|high>",
  "unique_approach": null,

  "title_match": {{
    "verdict":     "<full|partial|misleading>",
    "explanation": "<одно предложение>"
  }}
}}"""


def _run_educational_analysis(
    metadata: dict,
    segments: list[dict],
    chunks: list[dict],
    user_params: dict,
) -> dict:
    logger.info("LLM #2: educational analysis...")
    raw = _llm_call(
        system=_ANALYSIS_SYSTEM,
        user=_build_analysis_prompt(metadata, segments, chunks, user_params),
        temperature=0.0,
        model=NARRATIVE_LLM_MODEL_CREATIVE,   # Pro
        json_mode=True,
    )
    return json.loads(raw)


# ─── LLM #3: Adaptive Narrative ──────────────────────────────────────────────

_NARRATIVE_SYSTEM = (
    "Ты пишешь о видео от лица человека который искренне нашёл его ценным и понял почему. "
    "Стиль: спокойный, взрослый, естественный — как сильный студент или ассистент преподавателя "
    "который делится честным мнением, а не рекламирует. "
    "Передай что именно примечательно в этом видео — чем подход хорош, "
    "что объясняется лучше чем обычно, в чём его ценность для конкретного человека. "
    "Не просто перечисляй темы — показывай ценность. Без призывов, без преувеличений. "
    "Не используй сленг, шутки, блогерский тон, риторические вопросы. "
    "Запрещённые слова и обороты: 'короче', 'прикол', 'шаришь', 'врубиться', 'реально', "
    "'типа', 'ну тут', 'по сути тут', 'кайф', 'имба'. "
    "Только текст — никакого JSON, никаких заголовков."
)

_PERSON_BY_LEVEL = {
    "Школьник":                  "ученику старших классов",
    "Бакалавр (1-2 курс)":       "студенту младших курсов",
    "Бакалавр (3-4 курс)":       "студенту бакалавриата",
    "Магистр":                   "студенту магистратуры",
    "Специалист / Профессионал": "профессионалу в теме",
}

_SITUATION_BY_GOAL = {
    "Составить общее представление": "которому нужен быстрый обзор содержания",
    "Закрыть точечные вопросы":      "которому важно понять, где в видео разбирается нужная тема",
    "Последовательно изучить тему":  "который решает, стоит ли смотреть видео целиком",
}

_IMMERSION_SUFFIX = {
    "Изучаю с нуля":        "(только начинает разбираться)",
    "Знаю частично":        "(немного знаком с темой)",
    "Достаточно погружён":  "(в теме, просто не смотрел это видео)",
    "Экспертный уровень":   "(эксперт, интересует только что нового)",
}


def _get_social_context(user_params: dict) -> str:
    person    = _PERSON_BY_LEVEL.get(user_params.get("level", ""), "студенту")
    situation = _SITUATION_BY_GOAL.get(user_params.get("view_goal", ""), "")
    immersion = _IMMERSION_SUFFIX.get(user_params.get("immersion", ""), "")
    return f"{person} {situation} {immersion}".strip()


def _build_narrative_prompt(
    user_params: dict,
    segments: list[dict],
    analysis: dict,
) -> str:
    bloom_counts: dict[str, int] = {}
    for b in analysis.get("blooms_per_segment", []):
        bloom_counts[b["level"]] = bloom_counts.get(b["level"], 0) + 1
    dominant_bloom = max(bloom_counts, key=bloom_counts.get) if bloom_counts else "understand"

    high_prereqs = [
        p["concept"] for p in analysis.get("prerequisites", [])
        if p.get("confidence") == "high"
    ]

    topics = ", ".join(s["title"] for s in segments)
    gaps   = ", ".join(analysis.get("topic_coverage", {}).get("gaps", [])) or "нет"
    lp     = analysis.get("learning_path", {})
    tm     = analysis.get("title_match", {})

    return f"""Пользователь: {user_params.get('view_goal')}, уровень {user_params.get('level')}, погружённость «{user_params.get('immersion')}».

    Что известно о видео:
    - Темы: {topics}
    - Bloom's (доминантный): {dominant_bloom}
    - Пресреквизиты: {', '.join(high_prereqs) or 'не выявлены'}
    - Место в маршруте: {lp.get('type')} — {lp.get('reasoning')}
    - Пробелы в теме: {gaps}
    - Название: {tm.get('verdict')} — {tm.get('explanation')}
    - Уникальность: {analysis.get('unique_approach') or 'не обнаружена'}

    Напиши 2–3 предложения для {_get_social_context(user_params)}.

    Требования к стилю:
    - взрослый нейтральный пересказ, без панибратства;
    - не пытайся звучать как друг, блогер или школьник;
    - не используй сленг и эмоциональные усилители;
    - не пересказывай все темы списком, выдели главное;
    Сфокусируйся на том что делает именно это видео примечательным — 
    не просто перечисляй темы, а объясни в чём его ценность для этого человека.
    Не используй прямых призывов смотреть.
    """


def _run_narrative(
    user_params: dict,
    segments: list[dict],
    analysis: dict,
) -> str:
    logger.info("LLM #3: adaptive narrative...")
    return _llm_call(
        system=_NARRATIVE_SYSTEM,
        user=_build_narrative_prompt(user_params, segments, analysis),
        temperature=0.5,
        thinking=True,
        model=NARRATIVE_LLM_MODEL_CREATIVE,   # Pro
        json_mode=False,
    ).strip()


# ─── Assembly ─────────────────────────────────────────────────────────────────

def _assemble(
    chunks: list[dict],
    llm1_segments: list[dict],
    analysis: dict,
    narrative: str,
) -> NarrativeResult:
    id_to_start = {c["block_id"]: c["start"] for c in chunks}

    segments = [
        NarrativeSegment(
            title=seg["title"],
            description=seg.get("description", ""),
            start=id_to_start.get(seg.get("block_id_start", ""), 0.0),
        )
        for seg in llm1_segments
        if seg.get("block_id_start") in id_to_start
    ]

    blooms = [
        BloomsLabel(
            segment_title=b["title"],
            level=b["level"],
            evidence=b["evidence"],
        )
        for b in analysis.get("blooms_per_segment", [])
    ]

    bloom_counts: dict[str, int] = {}
    for b in blooms:
        bloom_counts[b.level] = bloom_counts.get(b.level, 0) + 1
    dominant_bloom = max(bloom_counts, key=bloom_counts.get) if bloom_counts else "understand"

    prereqs = [
        Prerequisite(concept=p["concept"], confidence=p["confidence"])
        for p in analysis.get("prerequisites", [])
    ]

    lp = analysis.get("learning_path", {})
    tc = analysis.get("topic_coverage", {})
    tm = analysis.get("title_match", {})

    return NarrativeResult(
        segments=segments,
        blooms_per_segment=blooms,
        blooms_dominant=dominant_bloom,
        prerequisites=prereqs,
        learning_path_type=lp.get("type", "standalone"),
        learning_path_reasoning=lp.get("reasoning", ""),
        topics_covered=tc.get("covered", []),
        topics_gaps=tc.get("gaps", []),
        info_density=analysis.get("info_density", "medium"),
        unique_approach=analysis.get("unique_approach"),
        title_match_verdict=tm.get("verdict", "partial"),
        title_match_explanation=tm.get("explanation", ""),
        narrative=narrative,
        n_chunks=len(chunks),
        n_segments=len(segments),
    )


# ─── Main entry point ─────────────────────────────────────────────────────────

def run(
    segments: list[dict],
    metadata: dict,
    user_params: dict,
) -> NarrativeResult:
    """
    Полный цикл: light_merge → три LLM-вызова.

    Args:
        segments:    whisper-сегменты [{"text", "start", "end"}, ...]
        metadata:    YouTube метаданные {"title", "description", "channel", ...}
        user_params: {"user_type", "level", "immersion", "view_goal"}
    """
    # 1. Лёгкое склеивание
    chunks = light_merge(segments)

    # 2. LLM #1 — LLM сама определяет тематические границы
    llm1_segments = _run_segment_extraction(metadata, chunks, user_params)
    logger.info(f"Тем найдено: {len(llm1_segments)}")

    # 3. LLM #2 — педагогический анализ
    analysis = _run_educational_analysis(metadata, llm1_segments, chunks, user_params)

    # 4. LLM #3 — адаптивный нарратив
    narrative_text = _run_narrative(user_params, llm1_segments, analysis)

    return _assemble(chunks, llm1_segments, analysis, narrative_text)
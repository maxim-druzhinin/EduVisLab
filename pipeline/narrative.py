"""
Модуль сегментации и адаптивного нарратива.

Шаги:
  1. light_merge      — whisper сегменты → ~50-секундные чанки
  2. BAAI/bge-m3      — семантические чанки с block_id
  3. LLM #1 (T=0.0)  — segment extraction: группировка в темы
  4. LLM #2 (T=0.0)  — educational analysis: Bloom's, пресреквизиты и т.д.
  5. LLM #3 (T=0.7)  — adaptive narrative: финальный текст для пользователя
"""

import json
import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
from openai import OpenAI
from sentence_transformers import SentenceTransformer

from config import (
    DEEPSEEK_API_KEY,
    NARRATIVE_LLM_MODEL,
    NARRATIVE_EMBED_MODEL,
    NARRATIVE_LIGHT_MERGE_SEC,
    NARRATIVE_MIN_SEGMENT_SEC,
    NARRATIVE_BOUNDARY_PERCENTILE,
)

logger = logging.getLogger(__name__)


# ─── Lazy singletons ──────────────────────────────────────────────────────────

_embed_model: Optional[SentenceTransformer] = None
_llm_client: Optional[OpenAI] = None


def _get_embed_model() -> SentenceTransformer:
    global _embed_model
    if _embed_model is None:
        logger.info(f"Загружаем эмбеддинговую модель: {NARRATIVE_EMBED_MODEL}")
        _embed_model = SentenceTransformer(NARRATIVE_EMBED_MODEL)
    return _embed_model


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
    learning_path_type: str        # intro / part_of_course / standalone / advanced
    learning_path_reasoning: str
    topics_covered: list[str]
    topics_gaps: list[str]
    info_density: str              # low / medium / high
    unique_approach: Optional[str]
    title_match_verdict: str       # full / partial / misleading
    title_match_explanation: str

    # Финальный текст
    narrative: str

    # Техническое
    n_semantic_chunks: int
    n_segments: int

    # Cosine similarities — переиспользуются для флага Хаотичность
    similarities: list[float]


# ─── Step 1: Light merge ──────────────────────────────────────────────────────

def light_merge(
    segments: list[dict],
    target_duration: float = NARRATIVE_LIGHT_MERGE_SEC,
) -> list[dict]:
    """
    Склеивает whisper-сегменты в блоки ~target_duration секунд.

    Входной формат:  [{"text": str, "start": float, "end": float}, ...]
    Выходной формат: [{"start": float, "text": str}, ...]
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
                "start": current_start,
                "text":  " ".join(current_text),
            })
            current_text = []
            if i + 1 < len(segments):
                current_start = segments[i + 1]["start"]

    logger.info(f"light_merge: {len(segments)} сегментов → {len(chunks)} чанков")
    return chunks


# ─── Step 2: Semantic chunking ────────────────────────────────────────────────

def _encode(texts: list[str]) -> np.ndarray:
    """Эмбеддинги через BGE-M3 (mean pooling, без префикса)."""
    return _get_embed_model().encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False,
    )


def find_semantic_boundaries(
    chunks: list[dict],
    embeddings: np.ndarray,
    threshold_percentile: float = NARRATIVE_BOUNDARY_PERCENTILE,
    min_duration: float = NARRATIVE_MIN_SEGMENT_SEC,
) -> tuple[list[int], list[float]]:
    """
    TextTiling через cosine similarity: ищет «долины» между соседними блоками.

    Возвращает (boundary_indices, similarities).
    similarities переиспользуется для флага Хаотичность.
    """
    similarities = [
        float(np.dot(embeddings[i], embeddings[i + 1]))
        for i in range(len(embeddings) - 1)
    ]

    threshold = float(np.percentile(similarities, threshold_percentile))
    boundaries = [0]

    for i, sim in enumerate(similarities):
        if sim < threshold:
            last_start = chunks[boundaries[-1]]["start"]
            if chunks[i + 1]["start"] - last_start >= min_duration:
                boundaries.append(i + 1)

    logger.info(
        f"Семантических границ: {len(boundaries)} "
        f"(threshold={threshold:.3f})"
    )
    return boundaries, similarities


def build_semantic_chunks(
    chunks: list[dict],
    boundaries: list[int],
) -> list[dict]:
    """
    Группирует чанки по границам, присваивает block_id.

    Выходной формат: [{"block_id": "B01", "start": float, "text": str}, ...]
    """
    result = []
    for idx, boundary in enumerate(boundaries):
        end = boundaries[idx + 1] if idx + 1 < len(boundaries) else len(chunks)
        block = chunks[boundary:end]
        result.append({
            "block_id": f"B{idx + 1:02d}",
            "start":    block[0]["start"],
            "text":     " ".join(c["text"] for c in block),
        })
    return result


# ─── Formatting helpers ───────────────────────────────────────────────────────

def _fmt_timestamp(seconds: float) -> str:
    m, s = int(seconds) // 60, int(seconds) % 60
    return f"{m}:{s:02d}"


def _fmt_duration(seconds: float) -> str:
    h = int(seconds) // 3600
    m = (int(seconds) % 3600) // 60
    s = int(seconds) % 60
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def _fmt_chunks_for_llm(semantic_chunks: list[dict]) -> str:
    lines = []
    for c in semantic_chunks:
        lines.append(f"[{c['block_id']} · {_fmt_timestamp(c['start'])}]\n{c['text']}")
    return "\n\n".join(lines)


# ─── LLM helper ───────────────────────────────────────────────────────────────

def _llm_call(
    system: str,
    user: str,
    temperature: float,
    thinking: bool,
    json_mode: bool = False,
    max_tokens: int = 1000,
) -> str:
    client = _get_llm_client()

    kwargs: dict = {
        "model":       NARRATIVE_LLM_MODEL,
        "messages":    [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        "temperature": temperature,
        "max_tokens":  max_tokens,
        "extra_body":  {
            "thinking": (
                {"type": "enabled", "budget_tokens": 2000}
                if thinking else
                {"type": "disabled"}
            ),
        },
    }

    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    response = client.chat.completions.create(**kwargs)
    return response.choices[0].message.content


# ─── LLM #1: Segment Extraction ───────────────────────────────────────────────

_SEGMENT_SYSTEM = (
    "Ты структурируешь транскрипт лекции на тематические части. "
    "Верни результат строго в JSON. Без текста вне JSON. Без markdown-блоков."
)


def _build_segment_prompt(metadata: dict, semantic_chunks: list[dict]) -> str:
    chapters_str = str(metadata.get("chapters")) if metadata.get("chapters") else "не указаны"
    return f"""=== МЕТАДАННЫЕ ===
Название: {metadata['title']}
Длительность: {_fmt_duration(metadata.get('duration', 0))}
Главы от автора: {chapters_str}

=== ТРАНСКРИПТ ===
{_fmt_chunks_for_llm(semantic_chunks)}

=== ЗАДАЧА ===
Сгруппируй блоки в тематические части. Один блок может принадлежать предыдущей
теме или начинать новую — решай по смыслу, а не по размеру блока.

{{
  "segments": [
    {{
      "block_id_start": "<block_id первого блока этой темы, строго из списка выше>",
      "title":          "<короткое название темы, 3-6 слов>",
      "description":    "<одно предложение о чём эта часть>"
    }}
  ]
}}"""


def _run_segment_extraction(
    metadata: dict,
    semantic_chunks: list[dict],
) -> list[dict]:
    logger.info("LLM #1: segment extraction...")
    raw = _llm_call(
        system=_SEGMENT_SYSTEM,
        user=_build_segment_prompt(metadata, semantic_chunks),
        temperature=0.0,
        thinking=False,
        json_mode=True,
        max_tokens=2500,
    )
    return json.loads(raw)["segments"]


# ─── LLM #2: Educational Analysis ────────────────────────────────────────────

_ANALYSIS_SYSTEM = (
    "Ты анализируешь образовательное видео с педагогической точки зрения. "
    "Верни результат строго в JSON. Без текста вне JSON. Без markdown-блоков."
)


def _build_analysis_prompt(
    metadata: dict,
    segments: list[dict],
    semantic_chunks: list[dict],
) -> str:
    segments_str = "\n".join(
        f"- {s['title']}: {s['description']}" for s in segments
    )
    return f"""=== МЕТАДАННЫЕ ===
Название: {metadata['title']}
Описание: {(metadata.get('description') or '')[:800]}
Канал: {metadata.get('channel') or 'не указан'}
Плейлист: {metadata.get('playlist') or 'не указан'}, позиция: {metadata.get('playlist_index') or 'не указана'}

=== ТЕМЫ ВИДЕО ===
{segments_str}

=== ПОЛНЫЙ ТРАНСКРИПТ ===
{_fmt_chunks_for_llm(semantic_chunks)}

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
    semantic_chunks: list[dict],
) -> dict:
    logger.info("LLM #2: educational analysis...")
    raw = _llm_call(
        system=_ANALYSIS_SYSTEM,
        user=_build_analysis_prompt(metadata, segments, semantic_chunks),
        temperature=0.0,
        thinking=False,
        json_mode=True,
        max_tokens=2500,
    )
    return json.loads(raw)


# ─── LLM #3: Adaptive Narrative ──────────────────────────────────────────────

_NARRATIVE_SYSTEM = (
    "Ты пишешь короткое описание образовательного видео для конкретного пользователя. "
    "Тон: разговорный, честный, как умный знакомый который посмотрел и рассказывает. "
    "Без канцелярщины и формальностей, но и без молодёжного сленга — "
    "никаких 'шаришь', 'врубиться', 'реально', 'крутишь'. "
    "Только текст — никакого JSON, никаких заголовков."
)

_PERSON_BY_LEVEL = {
    "Школьник":                  "однокласснику",
    "Бакалавр (1-2 курс)":       "одногруппнику с первого курса",
    "Бакалавр (3-4 курс)":       "одногруппнику",
    "Магистр":                   "коллеге по магистратуре",
    "Специалист / Профессионал": "коллеге по работе",
}

_SITUATION_BY_GOAL = {
    "Составить общее представление": "который спросил что смотришь",
    "Закрыть точечные вопросы":      "которому нужно найти где разбирают конкретную тему",
    "Последовательно изучить тему":  "который думает смотреть весь курс и спрашивает стоит ли",
}

_IMMERSION_SUFFIX = {
    "Изучаю с нуля":        "(только начинает разбираться)",
    "Знаю частично":        "(немного знаком с темой)",
    "Достаточно погружён":  "(в теме, просто не смотрел это видео)",
    "Экспертный уровень":   "(эксперт, интересует только что нового)",
}


def _get_social_context(user_params: dict) -> str:
    person    = _PERSON_BY_LEVEL.get(user_params.get("level", ""), "другу")
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

    topics  = ", ".join(s["title"] for s in segments)
    gaps    = ", ".join(analysis.get("topic_coverage", {}).get("gaps", [])) or "нет"
    lp      = analysis.get("learning_path", {})
    tm      = analysis.get("title_match", {})

    return f"""Пользователь: {user_params.get('view_goal')}, уровень {user_params.get('level')}, погружённость «{user_params.get('immersion')}».

Что известно о видео:
- Темы: {topics}
- Bloom's (доминантный): {dominant_bloom}
- Пресреквизиты: {', '.join(high_prereqs) or 'не выявлены'}
- Место в маршруте: {lp.get('type')} — {lp.get('reasoning')}
- Пробелы в теме: {gaps}
- Название: {tm.get('verdict')} — {tm.get('explanation')}
- Уникальность: {analysis.get('unique_approach') or 'не обнаружена'}

Напиши 2–3 предложения — ёмко и по делу, как объяснил бы {_get_social_context(user_params)}.
Учитывай цель просмотра."""


def _run_narrative(
    user_params: dict,
    segments: list[dict],
    analysis: dict,
) -> str:
    logger.info("LLM #3: adaptive narrative...")
    return _llm_call(
        system=_NARRATIVE_SYSTEM,
        user=_build_narrative_prompt(user_params, segments, analysis),
        temperature=0.7,
        thinking=False,
        json_mode=False,
        max_tokens=2500,
    ).strip()


# ─── Assembly ─────────────────────────────────────────────────────────────────

def _assemble(
    semantic_chunks: list[dict],
    similarities: list[float],
    llm1_segments: list[dict],
    analysis: dict,
    narrative: str,
) -> NarrativeResult:
    id_to_start = {c["block_id"]: c["start"] for c in semantic_chunks}

    segments = [
        NarrativeSegment(
            title=seg["title"],
            description=seg["description"],
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
        n_semantic_chunks=len(semantic_chunks),
        n_segments=len(segments),
        similarities=similarities,
    )


# ─── Main entry point ─────────────────────────────────────────────────────────

def run(
    segments: list[dict],
    metadata: dict,
    user_params: dict,
) -> NarrativeResult:
    """
    Полный цикл: семантическая сегментация + три LLM-вызова.

    Args:
        segments:    whisper-сегменты из TranscriptionResult
                     [{"text": str, "start": float, "end": float}, ...]
        metadata:    YouTube метаданные из fetch_metadata()
                     {"title", "description", "channel", "duration", ...}
        user_params: параметры пользователя
                     {"user_type", "level", "immersion", "view_goal"}

    Returns:
        NarrativeResult с сегментами, анализом и нарративом
    """
    # 1. Лёгкое склеивание для эмбеддинга
    chunks = light_merge(segments)

    # 2. Семантические чанки
    embeddings = _encode([c["text"] for c in chunks])
    boundaries, similarities = find_semantic_boundaries(chunks, embeddings)
    semantic_chunks = build_semantic_chunks(chunks, boundaries)
    logger.info(f"Семантических чанков: {len(semantic_chunks)}")

    # 3. LLM #1 — segment extraction
    llm1_segments = _run_segment_extraction(metadata, semantic_chunks)
    logger.info(f"Тем найдено: {len(llm1_segments)}")

    # 4. LLM #2 — educational analysis
    analysis = _run_educational_analysis(metadata, llm1_segments, semantic_chunks)

    # 5. LLM #3 — adaptive narrative
    narrative_text = _run_narrative(user_params, llm1_segments, analysis)

    return _assemble(semantic_chunks, similarities, llm1_segments, analysis, narrative_text)

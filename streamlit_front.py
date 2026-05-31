import time
import random
import math
import html
from urllib.parse import urlparse, parse_qs

import pandas as pd
import json

import streamlit as st
import streamlit.components.v1 as components

from pipeline.pipeline import run as run_pipeline


st.set_page_config(
    page_title="Анализ образовательного видео",
    page_icon="📊",
    layout="wide",
)

FONT_STACK = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif"

PAGE_BG = "#020817"
SIDEBAR_BG = "#071122"
SURFACE_BG = "#0b1220"
SURFACE_BG_ELEVATED = "#0f172a"
BORDER = "#1e293b"

TEXT_MAIN = "#e2e8f0"
TEXT_MUTED = "#94a3b8"
TEXT_SOFT = "#64748b"

TRACK_BG = "#1e293b"
TRACK_BG_SOFT = "#111827"

ACCENT_CYAN = "#38bdf8"
ACCENT_BLUE = "#60a5fa"
ACCENT_VIOLET = "#8b5cf6"
ACCENT_GREEN = "#22c55e"

DEMO_ORANGE = "#f97316"
DEMO_ORANGE_LIGHT = "#fb923c"

USER_TYPES = [
    "Обучающийся",
    "Спикер",
    "Продакшн",
    "Заказчик",
]

ROLE_ICONS = {
    "Обучающийся": "🎓",
    "Спикер": "🎤",
    "Продакшн": "🎬",
    "Заказчик": "📋",
}

LEVELS = [
    "Школьник",
    "Бакалавр (1-2 курс)",
    "Бакалавр (3-4 курс)",
    "Магистр",
    "Специалист / Профессионал",
]

IMMERSION_LEVELS = [
    "Изучаю с нуля",
    "Знаю частично",
    "Достаточно погружён",
    "Экспертный уровень",
]

VIEW_GOALS = [
    "Составить общее представление",
    "Закрыть точечные вопросы",
    "Последовательно изучить тему",
]

ROLE_INSTRUCTIONS = {
    "Обучающийся": {
        "start": (
            "Нарратив, темы с таймкодами, пререквизиты, "
            "пробелы покрытия и профиль подачи."
        ),
        "how": (
            "Проверь, подходит ли видео твоему уровню, цели просмотра "
            "и текущему запросу. Техническое качество и флаги восприятия "
            "смотри как возможные барьеры: будет ли видео удобно слушать, "
            "смотреть и понимать без лишнего напряжения."
        ),
    },

    "Спикер": {
        "start": (
            "Флаги восприятия и evidence по каждому "
            "сработавшему флагу."
        ),
        "how": (
            "Смотри, какие особенности речи, темпа, эмоциональности, "
            "жестикуляции и смысловой динамики могут мешать аудитории. "
            "Затем проверь профиль подачи: совпадает ли фактический стиль "
            "объяснения с тем форматом, который ты хотел получить — строгая "
            "лекция, практический разбор, обзор или tutorial."
        ),
    },

    "Продакшн": {
        "start": (
            "Техническое качество, визуальный шум, читаемость кадра, "
            "склейки и оформление доски/слайдов."
        ),
        "how": (
            "Отделяй проблемы записи и упаковки от проблем самого спикера. "
            "Сначала оцени звук, изображение, монтаж и визуальное оформление. "
            "Затем переходи к педагогическому анализу, структуре тем, "
            "пробелам покрытия и профилю подачи: они показывают, насколько "
            "методически собран материал и соответствует ли форма "
            "образовательной задаче."
        ),
    },

    "Заказчик": {
        "start": (
            "Покрытие темы, пробелы покрытия, соответствие названию, "
            "профиль подачи и нарративная оценка ценности."
        ),
        "how": (
            "Оцени, выполняет ли видео задачу, ради которой создавалось: "
            "подходит ли оно целевой аудитории, раскрывает ли заявленную тему "
            "и можно ли его публиковать, принимать или использовать в курсе. "
            "Техническое качество и флаги восприятия рассматривай как риски "
            "публикации, приёмки или дальнейшей доработки."
        ),
    },
}

MOCK_NARRATIVE = {
    "narrative": (
        "Это видео помогает увидеть цельную картину на стыке известных вам вещей "
        "из наивной теории множеств и строгого построения анализа. Его ценность не "
        "столько в списке тем, сколько в том, как плавно связываются операции над "
        "множествами и законы де Моргана с аксиоматическим определением действительных "
        "чисел — именно этот переход часто остаётся смазанным в стандартных курсах."
    ),
    "segments": [
        {"title": "Введение и история анализа",
         "description": "Предмет математического анализа как науки о приближениях, исторический обзор от Ньютона и Лейбница до строгого обоснования Коши, Кантора и Вейерштрасса.",
         "start": 6.3},
        {"title": "Множества и основные понятия функций",
         "description": "Вводятся наивная теория множеств, упорядоченная пара, декартово произведение, формальное определение функции, образа, прообраза, композиции и сужения.",
         "start": 463.2},
        {"title": "Классификация функций",
         "description": "Определяются инъективные, сюръективные и биективные отображения; вводится понятие обратной функции и разбираются примеры.",
         "start": 1391.1},
        {"title": "Индексированные семейства и законы де Моргана",
         "description": "Понятие семейства множеств, индексированного произвольным множеством; операции объединения и пересечения семейств; теорема де Моргана с доказательством.",
         "start": 2229.0},
        {"title": "Аксиоматическое определение действительных чисел",
         "description": "Аксиоматический подход к множеству R: алгебраические аксиомы поля, аксиомы линейного порядка, согласованного с операциями, и аксиома непрерывности, выделяющая R среди полей.",
         "start": 3400.1},
    ],
    "prerequisites": [
        {"concept": "Наивная теория множеств (принадлежность, включение, операции)", "confidence": "high"},
        {"concept": "Логические кванторы (для любого, существует)", "confidence": "medium"},
        {"concept": "Базовые алгебраические структуры (группа, поле)", "confidence": "low"},
    ],
    "learning_path": {
        "type": "intro",
        "reasoning": "Лекция вводит фундаментальные понятия математического анализа и аксиоматику действительных чисел, начиная с основ.",
    },
    "topic_coverage": {
        "covered": ["Введение и история анализа", "Множества и основные понятия функций",
                    "Классификация функций", "Индексированные семейства и законы де Моргана",
                    "Аксиоматическое определение действительных чисел"],
        "gaps": [],
    },
    "info_density": "high",
    "title_match": {
        "verdict": "full",
        "explanation": "Название точно отражает содержание: введение в анализ и подробное обсуждение действительных чисел.",
    },
}


DEMO_RESULTS_CSV = "validation_dataset/pipeline_results.csv"
DEMO_NARRATIVE_CSV = "validation_dataset/pipeline_narrative.csv"


def _safe_json_loads(value, default):
    if value is None or pd.isna(value):
        return default
    if isinstance(value, (list, dict)):
        return value
    try:
        return json.loads(value)
    except Exception:
        return default


def load_demo_results():
    results_df = pd.read_csv(DEMO_RESULTS_CSV)
    narrative_df = pd.read_csv(DEMO_NARRATIVE_CSV)

    narrative_by_url = {
        row["meta.url"]: row
        for _, row in narrative_df.iterrows()
    }

    demo_results = []

    for i, row in results_df.iterrows():
        video_url = row.get("meta.video_url", "")
        narrative_row = narrative_by_url.get(video_url)

        if narrative_row is not None:
            narrative = {
                "narrative": narrative_row.get("narrative.narrative", ""),
                "segments": _safe_json_loads(narrative_row.get("narrative.segments"), []),
                "prerequisites": _safe_json_loads(narrative_row.get("narrative.prerequisites"), []),
                "learning_path": {
                    "type": narrative_row.get("narrative.learning_path.type", ""),
                    "reasoning": narrative_row.get("narrative.learning_path.reasoning", ""),
                },
                "topic_coverage": {
                    "covered": _safe_json_loads(narrative_row.get("narrative.topic_coverage.covered"), []),
                    "gaps": _safe_json_loads(narrative_row.get("narrative.topic_coverage.gaps"), []),
                },
                "info_density": narrative_row.get("narrative.info_density", ""),
                "title_match": {
                    "verdict": narrative_row.get("narrative.title_match.verdict", ""),
                    "explanation": narrative_row.get("narrative.title_match.explanation", ""),
                },
            }
        else:
            narrative = MOCK_NARRATIVE

        academic_score = float(row.get("delivery_profile.academic_score", 0.5) or 0.5)
        instrumental_score = float(row.get("delivery_profile.instrumental_score", 0.5) or 0.5)

        metrics = {
            "academic_score": academic_score,
            "instrumental_score": instrumental_score,
            "tech_quality": round(float(row.get("technical_quality.score", 0) or 0), 1),
            "audio_score": row.get("technical_quality.audio_score", None),
            "video_score": row.get("technical_quality.video_score", None),
        }

        result = {
            "id": i + 1,
            "created_at": str(row.get("meta.processed_at", "")),
            "video_url": video_url,
            "video_title": row.get("meta.youtube.title", f"Видео #{i + 1}"),
            "user_type": "Обучающийся",
            "audience_level": "Бакалавр (1-2 курс)",
            "immersion_level": "Знаю частично",
            "view_goal": "Составить общее представление",
            "summary": narrative.get("narrative", ""),
            "metrics": metrics,
            "narrative": narrative,
            "warnings": build_warnings_from_row(row),
        }

        demo_results.append(result)

    return demo_results



def build_result_from_pipeline_output(pipeline_output: dict, submitted_data: dict) -> dict:
    meta = pipeline_output.get("meta", {}) or {}
    youtube = meta.get("youtube", {}) or {}
    technical = pipeline_output.get("technical_quality", {}) or {}
    delivery = pipeline_output.get("delivery_profile", {}) or {}
    narrative = pipeline_output.get("narrative", {}) or {}

    metrics = {
        "academic_score": float(delivery.get("academic_score", 0.5) or 0.5),
        "instrumental_score": float(delivery.get("instrumental_score", 0.5) or 0.5),
        "tech_quality": float(technical.get("score", 0) or 0),
        "audio_score": technical.get("audio_score"),
        "video_score": technical.get("video_score"),
    }

    return {
        "id": st.session_state.result_counter + 1,
        "created_at": str(meta.get("processed_at", time.strftime("%Y-%m-%d %H:%M:%S"))),
        "video_url": meta.get("video_url", submitted_data.get("video_url", "")),
        "video_title": youtube.get("title") or mock_video_title(submitted_data.get("video_url", ""), st.session_state.result_counter + 1),
        "user_type": submitted_data.get("user_type", ""),
        "audience_level": submitted_data.get("audience_level", ""),
        "immersion_level": submitted_data.get("immersion_level", ""),
        "view_goal": submitted_data.get("view_goal", ""),
        "summary": narrative.get("narrative", ""),
        "metrics": metrics,
        "narrative": narrative,
        "warnings": build_warnings_from_pipeline_output(pipeline_output),
    }


def build_warnings_from_pipeline_output(pipeline_output: dict):
    flags = pipeline_output.get("jung_flags", {}) or {}

    def flag_active(name: str) -> bool:
        data = flags.get(name, {}) or {}
        return _csv_bool(data.get("flag", False))

    return [
        {
            "title": "Аудио дисбаланс",
            "description": (
                "Речь или звук мешают восприятию: заметные особенности голоса, "
                "тембра, фоновые шумы, слова-паразиты или просторечия."
            ),
            "active": flag_active("logic"),
            "icon_svg": WARNING_ICON_CHAOS,
        },
        {
            "title": "Визуальный шум",
            "description": (
                "Избыток визуальных стимулов, мешающих воспринимать содержание: "
                "резкие склейки, хаотичное движение в кадре или перегруженная сцена."
            ),
            "active": flag_active("sensor"),
            "icon_svg": WARNING_ICON_VISUAL_NOISE,
        },
        {
            "title": "Смысловая унылость",
            "description": (
                "Монотонность подачи и бедность идей: однообразная лексика, "
                "факты без обобщений и связей, затянутое вступление."
            ),
            "active": flag_active("intuitive"),
            "icon_svg": WARNING_ICON_MONOTONY,
        },
        {
            "title": "Эмоциональный дисбаланс",
            "description": (
                "Несоответствие эмоционального тона контексту: чрезмерная "
                "или наоборот полностью отсутствующая эмоциональная окраска речи."
            ),
            "active": flag_active("emotion"),
            "icon_svg": WARNING_ICON_PLACEHOLDER,
        },
    ]




WARNING_ICON_CHAOS = """
<svg viewBox="0 0 70 70" fill="none" xmlns="http://www.w3.org/2000/svg">
  <g transform="translate(35,35) scale(1.2) translate(-58,-58)">
    <path d="M30.6445 50.1478V66.8655H41.7896L55.721 80.7968V36.2165L41.7896 50.1478H30.6445Z"
          stroke="currentColor" stroke-width="3.6" stroke-linecap="round" stroke-linejoin="round"/>
    <path d="M40.3964 64.9151C40.8581 64.9151 41.2323 64.5409 41.2323 64.0792C41.2323 63.6176 40.8581 63.2433 40.3964 63.2433C39.9348 63.2433 39.5605 63.6176 39.5605 64.0792C39.5605 64.5409 39.9348 64.9151 40.3964 64.9151Z"
          fill="currentColor" stroke="currentColor" stroke-width="3.6" stroke-linecap="round" stroke-linejoin="round"/>
    <path d="M64.0801 44.5753L68.2595 51.541L64.0801 58.5067L68.2595 65.4724L64.0801 72.438"
          stroke="currentColor" stroke-width="3.6" stroke-linecap="round" stroke-linejoin="round"/>
    <path d="M72.4395 41.7891L76.6189 50.1479L72.4395 58.5067L76.6189 66.8655L72.4395 75.2243"
          stroke="currentColor" stroke-width="3.6" stroke-linecap="round" stroke-linejoin="round"/>
  </g>
</svg>
"""

WARNING_ICON_VISUAL_NOISE = """
<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none"
stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round">
  <path d="M0 0h24v24H0z" stroke="none"/>
  <path d="M15.03 17.478A8.8 8.8 0 0 1 12 18q-5.4 0-9-6 3.6-6 9-6t9 6a21 21 0 0 1-.258.419M19 16v3m0 3v.01"/>
  <path d="m12 9-2 3h4l-2 3"/>
</svg>
"""

WARNING_ICON_MONOTONY = """
<svg width="200px" height="200px" viewBox="0 0 200 200" xmlns="http://www.w3.org/2000/svg">
  <path d="M118,26.7c-8-2.5-13,7-9.5,14.5a57.43,57.43,0,0,1,6,26.5c0,31.5-26,57.5-58,57.5a74.59,74.59,0,0,1-10.5-1c-9-1.5-19,6-14,14a76,76,0,0,0,64.5,35.5c42,0,76-33.5,76-75,0-34-24-62.5-54.5-72Zm-21.5,127a59.19,59.19,0,0,1-31-9,77.43,77.43,0,0,0,68.5-77,60.51,60.51,0,0,0-.5-9.5A53.44,53.44,0,0,1,152,98.7C152.5,128.7,127.5,153.7,96.5,153.7ZM48,58.7l-13.5,19a9.84,9.84,0,0,0,8,15.5h35a10,10,0,0,0,0-20H62a54.58,54.58,0,0,1,6.5-8.5c3-4.5,8.5-9.5,9.5-14.5,1-3.5-1-7.5-4-9.5-5-3.5-13.5-2-19-2H37.5a10,10,0,0,0,0,20Z" fill="currentColor"/>
</svg>
"""

WARNING_ICON_PLACEHOLDER = """
<svg viewBox="0 0 512 512" xmlns="http://www.w3.org/2000/svg">
  <path fill="currentColor" d="M480.037,86.769c-15.324-23.21-41.626-38.564-71.459-38.564c-67.509,0-125.842,39.996-152.577,97.531
      c-26.735-57.535-85.068-97.531-152.577-97.531c-29.833,0-56.136,15.353-71.459,38.564C13.659,91.123,0,107.6,0,127.217
      c0,22.93,18.655,41.584,41.585,41.584c7.359,0,14.272-1.93,20.274-5.297c1.734,1.754,3.349,3.655,4.824,5.699
      c7.63,10.583,10.517,23.475,8.13,36.303c-0.094,0.507-0.16,1.011-0.203,1.513c-15.268,6.161-26.075,21.124-26.075,38.573
      c0,17.426,10.779,32.373,26.015,38.549v127.465c0,28.776,23.411,52.185,52.186,52.185h260.188
      c28.776,0,52.186-23.411,52.186-52.185v-127.58c15.081-6.248,25.719-21.119,25.719-38.433c0-17.994-11.489-33.349-27.515-39.129
      c-0.039-0.319-0.069-0.637-0.129-0.957c-2.387-12.827,0.499-25.719,8.13-36.303c1.474-2.045,3.09-3.945,4.824-5.7
      c6.002,3.367,12.916,5.297,20.275,5.297c22.93,0,41.584-18.655,41.584-41.584C512,107.6,498.341,91.123,480.037,86.769z
      M41.585,137.663c-5.761,0.001-10.447-4.685-10.447-10.446c0-5.761,4.686-10.446,10.447-10.446
      c5.76,0,10.446,4.686,10.446,10.446C52.031,132.977,47.345,137.663,41.585,137.663z M258.738,235.148
      c5.761,0,10.447,4.686,10.447,10.446c0,5.76-4.687,10.446-10.447,10.446c-5.76,0.001-10.446-4.685-10.446-10.446
      C248.292,239.834,252.978,235.148,258.738,235.148z M66.43,93.889c9.719-9.019,22.722-14.547,36.994-14.547
      c72.874,0,132.632,57.191,136.77,129.046c-3.558,1.781-6.825,4.054-9.713,6.733c-8.19,7.6-13.328,18.445-13.328,30.473
      c0,8.048,2.305,15.564,6.28,21.937l-11.366,9.434l-35.811,29.724l-31.235-27.761l-17.37-15.438
      c2.597-5.423,4.054-11.492,4.054-17.896c0-15.55-8.585-29.126-21.258-36.258c-1.412-0.794-2.871-1.513-4.378-2.141
      c2.684-19.936-2.267-39.75-14.126-56.2c-3.143-4.359-6.67-8.327-10.514-11.883c1.128-3.771,1.743-7.762,1.743-11.894
      C83.171,113.592,76.585,101.478,66.43,93.889z M90.12,256.04c-5.76,0-10.446-4.686-10.446-10.446s4.685-10.446,10.446-10.446
      s10.447,4.686,10.447,10.446S95.881,256.04,90.12,256.04z M407.973,411.61c0,11.605-9.442,21.047-21.048,21.047H126.737
      c-11.606,0-21.047-9.442-21.047-21.047v-0.564h302.283V411.61z M407.974,379.907H105.689v-94.275l59.877,53.216
      c5.75,5.11,14.368,5.254,20.287,0.343l63.865-53.009c2.906,0.646,5.922,0.996,9.021,0.996c2.736,0,5.409-0.273,7.998-0.78
      l59.013,52.448c2.945,2.617,6.642,3.932,10.344,3.932c3.525,0,7.056-1.193,9.942-3.589l61.937-51.408V379.907z
      M423.245,256.041c-5.761,0-10.446-4.686-10.446-10.446s4.686-10.446,10.446-10.446s10.446,4.686,10.446,10.446
      S429.005,256.041,423.245,256.041z M428.831,127.217c0,4.133,0.615,8.123,1.742,11.894c-3.846,3.557-7.371,7.524-10.515,11.883
      c-11.972,16.606-16.892,36.639-14.038,56.765c-1.014,0.464-2.008,0.963-2.977,1.504c-12.743,7.114-21.383,20.729-21.383,36.33
      c0,7.049,1.769,13.689,4.876,19.512l-8.431,6.998l-41.667,34.584l-36.343-32.301l-6.656-5.915
      c4.345-6.568,6.884-14.432,6.884-22.878c0-12.158-5.246-23.113-13.592-30.724c-4.224-3.853-9.248-6.837-14.784-8.697
      c5.225-70.811,64.507-126.83,136.63-126.83c14.273,0,27.275,5.528,36.994,14.547C435.417,101.479,428.831,113.594,428.831,127.217z
      M470.415,137.663c-5.76,0-10.446-4.685-10.446-10.446c0-5.761,4.686-10.446,10.446-10.446s10.446,4.686,10.446,10.446
      C480.861,132.977,476.174,137.663,470.415,137.663z"/>
</svg>
"""

def render_warning_flags(flags: list[dict], columns: int = 2):
    cards_html = ""

    for i, flag in enumerate(flags):
        active = flag.get("active", False)
        title = html.escape(flag.get("title", "Флаг"))
        description = html.escape(flag.get("description", "Описание пока не задано."))
        icon_svg = flag.get("icon_svg", WARNING_ICON_PLACEHOLDER)

        if active:
            border = "rgba(249, 115, 22, 0.42)"
            icon_color = "#fb923c"
            status_bg = "rgba(249, 115, 22, 0.18)"
            status_color = "#fdba74"
            glow = (
                "0 0 0 1px rgba(249, 115, 22, 0.16), "
                "0 10px 24px rgba(0, 0, 0, 0.28), "
                "0 0 16px rgba(249, 115, 22, 0.12), "
                "0 0 28px rgba(251, 146, 60, 0.08)"
            )
            badge_text = "обнаружено"
            icon_bg = "rgba(249, 115, 22, 0.08)"
            icon_border = "rgba(249, 115, 22, 0.16)"
        else:
            border = "rgba(59, 130, 246, 0.14)"
            icon_color = "#7c6a4a"
            status_bg = "rgba(148, 163, 184, 0.08)"
            status_color = "#94a3b8"
            glow = (
                "0 0 0 1px rgba(59, 130, 246, 0.08), "
                "0 8px 18px rgba(0, 0, 0, 0.18)"
            )
            badge_text = "не активно"
            icon_bg = SURFACE_BG_ELEVATED
            icon_border = "rgba(148, 163, 184, 0.08)"

        position_class = "expand-right" if i % 2 == 0 else "expand-left"

        cards_html += f"""
        <input type="checkbox" id="warn-toggle-{i}" class="warn-toggle">
        <label for="warn-toggle-{i}" class="warn-card {position_class} {'active' if active else 'inactive'}" style="border-color:{border}; box-shadow:{glow};">
            <div class="warn-face warn-front">
                <div class="warn-icon big" style="color:{icon_color}; background:{icon_bg}; border-color:{icon_border};">
                    {icon_svg}
                </div>
            </div>

            <div class="warn-face warn-back">
                <div class="warn-back-inner">
                    <div class="warn-back-title">{title}</div>
                    <!-- <div class="warn-back-badge" style="background:{status_bg}; color:{status_color};">{badge_text}</div> -->
                    <div class="warn-back-description">{description}</div>
                </div>
            </div>
        </label>
        """

    html_block = f"""
    <!DOCTYPE html>
    <html>
    <head>
    <meta charset="utf-8">
    <style>
        html, body {{
            margin: 0;
            padding: 0;
            background: transparent;
            font-family: {FONT_STACK};
            overflow: visible;
        }}

        .warn-grid {{
            display: grid;
            grid-template-columns: repeat({columns}, 1fr);
            gap: 15px 25px;
            padding: 15px 5px 0 5px;
            height: 100%;
            align-content: center;
            justify-content: center;
            overflow: visible;
        }}

        .warn-card {{
            position: relative;
            display: block;
            height: 115px;
            width: 100%;
            border-radius: 20px;
            background: {SURFACE_BG};
            overflow: hidden;
            box-sizing: border-box;
            cursor: pointer;
            z-index: 1;
            transition:
                transform 0.18s ease,
                box-shadow 0.18s ease,
                border-color 0.18s ease,
                width 0.22s ease,
                min-width 0.22s ease,
                margin 0.22s ease;
        }}

        .warn-toggle {{
            display: none;
        }}

        .warn-card:hover {{
            transform: translateY(-1px);
        }}

        .warn-face {{
            position: absolute;
            inset: 0;
            transition: opacity 0.2s ease, transform 0.2s ease;
        }}

        .warn-front {{
            display: flex;
            align-items: center;
            justify-content: center;
            opacity: 1;
            transform: scale(1);
        }}

        .warn-back {{
            opacity: 0;
            transform: scale(0.97);
            padding: 12px;
            box-sizing: border-box;
        }}

        .warn-toggle:checked + .warn-card {{
            z-index: 20;
        }}

        .warn-toggle:checked + .warn-card.expand-right {{
            width: calc(100% + 138px);
            min-width: calc(100% + 138px);
            transform: translateX(0);
        }}

        .warn-toggle:checked + .warn-card.expand-left {{
            width: calc(100% + 138px);
            min-width: calc(100% + 138px);
            transform: translateX(-138px);
        }}

        .warn-toggle:checked + .warn-card .warn-front {{
            opacity: 0;
            transform: scale(0.96);
            pointer-events: none;
        }}

        .warn-toggle:checked + .warn-card .warn-back {{
            opacity: 1;
            transform: scale(1);
        }}

        .warn-icon.big {{
            width: 70px;
            height: 70px;
            min-width: 60px;
            display: flex;
            align-items: center;
            justify-content: center;
            border-radius: 18px;
            border: 1px solid;
            box-sizing: border-box;
        }}

        .warn-icon.big svg {{
            width: 40px;
            height: 40px;
            display: block;
        }}

        .warn-back-inner {{
            display: flex;
            flex-direction: column;
            height: 100%;
        }}

        .warn-back-title {{
            color: {TEXT_MAIN};
            font-size: 0.88rem;
            font-weight: 700;
            line-height: 1.15;
            margin-bottom: 6px;
            padding-right: 4px;
        }}

        .warn-back-badge {{
            display: inline-block;
            align-self: flex-start;
            padding: 3px 8px;
            border-radius: 999px;
            font-size: 0.68rem;
            font-weight: 600;
            line-height: 1;
            margin-bottom: 8px;
        }}

        .warn-back-description {{
            color: {TEXT_MUTED};
            font-size: 0.76rem;
            line-height: 1.26;
            overflow: hidden;
            display: -webkit-box;
            -webkit-line-clamp: 5;
            -webkit-box-orient: vertical;
            padding-right: 4px;
        }}
    </style>
    </head>
    <body>
        <div class="warn-grid">
            {cards_html}
        </div>
    </body>
    </html>
    """
    render_html_block(html_block, height=280, width=None)


def build_warning_flags(metrics: dict):
    return [
        {
            "title": "Аудио дисбаланс",
            "description": (
                "Речь или звук мешают восприятию: заметные особенности голоса, "
                "тембра, фоновые шумы, слова-паразиты или просторечия."
            ),
            "active": True,
            "icon_svg": WARNING_ICON_CHAOS,
        },
        {
            "title": "Визуальный шум",
            "description": (
                "Монотонность подачи и бедность идей: однообразная лексика, "
                "факты без обобщений и связей, затянутое вступление."
            ),
            "active": False,
            "icon_svg": WARNING_ICON_VISUAL_NOISE,
        },
        {
            "title": "Смысловая унылость",
            "description": (
                "Монотонность подачи и бедность идей: однообразная лексика, "
                "факты без обобщений и связей, затянутое вступление."
            ),
            "active": True,
            "icon_svg": WARNING_ICON_MONOTONY,
        },
        {
            "title": "Эмоциональный дисбаланс",
            "description": (
                "Несоответствие эмоционального тона контексту: чрезмерная "
                "или наоборот полностью отсутствующая эмоциональная окраска речи."
            ),
            "active": True,
            "icon_svg": WARNING_ICON_PLACEHOLDER,
        },
    ]

def _csv_bool(value) -> bool:
    if pd.isna(value):
        return False
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "да"}


def build_warnings_from_row(row):
    return [
        {
            "title": "Аудио дисбаланс",
            "description": (
                "Речь или звук мешают восприятию: заметные особенности голоса, "
                "тембра, фоновые шумы, слова-паразиты или просторечия."
            ),
            "active": _csv_bool(row.get("jung_flags.logic.flag", False)),
            "icon_svg": WARNING_ICON_CHAOS,
        },
        {
            "title": "Визуальный шум",
            "description": (
                "Избыток визуальных стимулов, мешающих воспринимать содержание: "
                "резкие склейки, хаотичное движение в кадре или перегруженная сцена."
            ),
            "active": _csv_bool(row.get("jung_flags.sensor.flag", False)),
            "icon_svg": WARNING_ICON_VISUAL_NOISE,
        },
        {
            "title": "Смысловая унылость",
            "description": (
                "Монотонность подачи и бедность идей: однообразная лексика, "
                "факты без обобщений и связей, затянутое вступление."
            ),
            "active": _csv_bool(row.get("jung_flags.intuitive.flag", False)),
            "icon_svg": WARNING_ICON_MONOTONY,
        },
        {
            "title": "Эмоциональный дисбаланс",
            "description": (
                "Несоответствие эмоционального тона контексту: чрезмерная "
                "или наоборот полностью отсутствующая эмоциональная окраска речи."
            ),
            "active": _csv_bool(row.get("jung_flags.emotion.flag", False)),
            "icon_svg": WARNING_ICON_PLACEHOLDER,
        },
    ]


def inject_global_styles():
    st.markdown(
        f"""
        <style>
        html, body {{
            font-family: {FONT_STACK};
        }}

        .stApp {{
            background: {PAGE_BG};
            color: {TEXT_MAIN};
            font-family: {FONT_STACK};
        }}

        [data-testid="stAppViewContainer"] {{
            background: {PAGE_BG};
        }}

        [data-testid="stHeader"] {{
            background: transparent;
        }}

        [data-testid="stMainBlockContainer"] {{
            padding-top: 2rem;
        }}

        h1, h2, h3, h4, h5, h6 {{
            color: {TEXT_MAIN} !important;
            font-family: {FONT_STACK} !important;
        }}

        p, li, label, input, textarea, button {{
            font-family: {FONT_STACK} !important;
        }}

        .stMarkdown, .stMarkdown p, .stMarkdown li, .stCaption {{
            color: {TEXT_MUTED};
        }}

        [data-testid="stForm"] {{
            background: {SURFACE_BG};
            border: 1px solid {BORDER};
            border-radius: 24px;
            padding: 1rem 1rem 0.75rem 1rem;
        }}

        .stButton > button[kind="primary"] {{
            background: linear-gradient(135deg, #f97316 0%, #fb923c 100%) !important;
            color: #ffffff !important;
            border: 1px solid rgba(251,146,60,0.45) !important;
            box-shadow:
                0 0 18px rgba(249,115,22,0.28),
                0 10px 24px rgba(0,0,0,0.30) !important;
        }}

        .stTextInput > div > div > input {{
            background: {SURFACE_BG_ELEVATED};
            color: {TEXT_MAIN};
            border: 1px solid {BORDER};
            border-radius: 12px;
            font-family: {FONT_STACK};
        }}

        .stTextInput > div > div > input::placeholder {{
            color: {TEXT_SOFT};
        }}

        .stTextInput label,
        .stRadio label,
        .stCheckbox label {{
            color: {TEXT_MAIN} !important;
        }}

        .stRadio div[role="radiogroup"] label,
        .stCheckbox label p {{
            color: {TEXT_MAIN} !important;
        }}

        .stRadio [data-baseweb="radio"] > div {{
            color: {TEXT_MAIN} !important;
        }}

        .stCheckbox [data-baseweb="checkbox"] > div {{
            color: {TEXT_MAIN} !important;
        }}

        .stAlert {{
            background: {SURFACE_BG_ELEVATED};
            border: 1px solid {BORDER};
            border-radius: 16px;
            color: {TEXT_MAIN};
        }}

        .stButton > button,
        .stForm button {{
            background: {SURFACE_BG_ELEVATED};
            color: {TEXT_MAIN};
            border: 1px solid {BORDER};
            border-radius: 14px;
            font-weight: 600;
            font-family: {FONT_STACK};
        }}

        .stButton > button:hover,
        .stForm button:hover {{
            border-color: {ACCENT_VIOLET};
            color: #ffffff;
            background: #111c31;
            box-shadow:
                0 0 0 1px rgba(139, 92, 246, 0.45),
                0 0 24px rgba(139, 92, 246, 0.35),
                0 0 56px rgba(59, 130, 246, 0.20),
                0 12px 30px rgba(0, 0, 0, 0.45) !important;
        }}

        .stButton > button:focus,
        .stForm button:focus {{
            box-shadow: none;
            border-color: {ACCENT_VIOLET};
        }}

        [data-testid="stForm"],
        [data-testid="stExpander"],
        [data-testid="stVideo"] {{
            box-shadow:
                0 0 0 1px rgba(59, 130, 246, 0.18),
                0 16px 40px rgba(0, 0, 0, 0.45),
                0 0 36px rgba(59, 130, 246, 0.16),
                0 0 80px rgba(139, 92, 246, 0.10) !important;
        }}

        

        [data-testid="stExpander"] {{
            background: {SURFACE_BG};
            border: 1px solid {BORDER};
            border-radius: 18px;
            overflow: hidden;
        }}

        [data-testid="stExpander"] details {{
            background: {SURFACE_BG};
        }}

        [data-testid="stExpander"] details summary {{
            background: {SURFACE_BG};
            color: {TEXT_MAIN};
        }}

        [data-testid="stExpander"] details summary p {{
            color: {TEXT_MAIN} !important;
            font-family: {FONT_STACK} !important;
            margin: 0 !important;
        }}

        /* Возвращаем иконный шрифт стрелке Streamlit */
        [data-testid="stExpander"] details summary .material-symbols-rounded {{
            font-family: "Material Symbols Rounded" !important;
            font-weight: normal !important;
            font-style: normal !important;
            font-size: 20px !important;
            line-height: 1 !important;
            letter-spacing: normal !important;
            text-transform: none !important;
            white-space: nowrap !important;
            direction: ltr !important;
            -webkit-font-smoothing: antialiased !important;
            font-variation-settings: 'FILL' 0, 'wght' 400, 'GRAD' 0, 'opsz' 24 !important;
        }}

        [data-testid="stExpander"] details summary .material-symbols-outlined {{
            font-family: "Material Symbols Outlined" !important;
            font-weight: normal !important;
            font-style: normal !important;
            font-size: 20px !important;
            line-height: 1 !important;
            letter-spacing: normal !important;
            text-transform: none !important;
            white-space: nowrap !important;
            direction: ltr !important;
            -webkit-font-smoothing: antialiased !important;
            font-variation-settings: 'FILL' 0, 'wght' 400, 'GRAD' 0, 'opsz' 24 !important;
        }}

        /* На случай, если стрелка рендерится через span без класса */
        [data-testid="stExpander"] details summary span[aria-hidden="true"] {{
            font-family: "Material Symbols Rounded" !important;
            font-weight: normal !important;
            font-style: normal !important;
            font-size: 20px !important;
            line-height: 1 !important;
            letter-spacing: normal !important;
            text-transform: none !important;
            white-space: nowrap !important;
            direction: ltr !important;
            -webkit-font-smoothing: antialiased !important;
        }}

        [data-testid="stVideo"] {{
            background: {SURFACE_BG};
            border: 1px solid {BORDER};
            border-radius: 20px;
            padding: 8px;
        }}

        [data-testid="stProgressBar"] > div {{
            background-color: {TRACK_BG_SOFT};
            border-radius: 999px;
        }}

        [data-testid="stProgressBar"] div[role="progressbar"] {{
            background-color: {ACCENT_VIOLET};
        }}

        hr {{
            border-color: {BORDER} !important;
        }}

        section[data-testid="stSidebar"] {{
            background: {SIDEBAR_BG};
            border-right: 1px solid {BORDER};
        }}

        section[data-testid="stSidebar"] * {{
            color: {TEXT_MAIN};
            font-family: {FONT_STACK};
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

def render_html_block(html_content: str, height: int = 240, width: int | None = None):
    kwargs = {
        "html": html_content,
        "height": height,
        "scrolling": False,
    }
    if width is not None:
        kwargs["width"] = width
    components.html(**kwargs)


def render_bipolar_watch_widget(
    value: int,
    title: str = "Профиль подачи",
    left_label: str = "Инструмент",
    right_label: str = "Академия",
    size: int = 280,
    academic_score: float = 0.5,
    instrumental_score: float = 0.5,
):
    value = max(-100, min(100, value))
    normalized = (value + 100) / 200
    marker_offset = normalized * 100

    grad_id = f"grad_bipolar_{abs(value)}_{random.randint(1000,9999)}"
    widget_id = f"bipolar_{random.randint(100000,999999)}"
    info_id = f"info_{random.randint(100000,999999)}"

    value_text = str(abs(value))

    def hex_to_rgb(h):
        h = h.lstrip("#")
        return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

    def rgb_to_hex(rgb):
        return "#{:02x}{:02x}{:02x}".format(*rgb)

    def lerp(a, b, t):
        return int(round(a + (b - a) * t))

    def blend(c1, c2, t):
        r1, g1, b1 = hex_to_rgb(c1)
        r2, g2, b2 = hex_to_rgb(c2)
        return rgb_to_hex((lerp(r1, r2, t), lerp(g1, g2, t), lerp(b1, b2, t)))

    t = normalized
    if t <= 0.18:
        marker_fill = blend("#ea580c", "#f59e0b", t / 0.18)
    elif t <= 0.36:
        marker_fill = blend("#f59e0b", "#fbbf24", (t - 0.18) / 0.18)
    elif t <= 0.46:
        marker_fill = blend("#fbbf24", "#fde68a", (t - 0.36) / 0.10)
    elif t <= 0.50:
        marker_fill = blend("#fde68a", "#f8fafc", (t - 0.46) / 0.04)
    elif t <= 0.54:
        marker_fill = blend("#f8fafc", "#ddd6fe", (t - 0.50) / 0.04)
    elif t <= 0.64:
        marker_fill = blend("#ddd6fe", "#c4b5fd", (t - 0.54) / 0.10)
    elif t <= 0.82:
        marker_fill = blend("#c4b5fd", "#8b5cf6", (t - 0.64) / 0.18)
    else:
        marker_fill = blend("#8b5cf6", "#6d28d9", (t - 0.82) / 0.18)

    PLOT_X0, PLOT_X1 = 32, 188
    PLOT_Y0, PLOT_Y1 = 40, 188
    PLOT_W = PLOT_X1 - PLOT_X0
    PLOT_H = PLOT_Y1 - PLOT_Y0
    PLOT_MID_X = PLOT_X0 + PLOT_W / 2
    PLOT_MID_Y = PLOT_Y0 + PLOT_H / 2

    dot_x = PLOT_X0 + instrumental_score * PLOT_W
    dot_y = PLOT_Y1 - academic_score * PLOT_H

    html_block = f"""
    <!DOCTYPE html>
    <html>
    <head>
    <meta charset="utf-8">
    <style>
        html, body {{
            margin: 0;
            background: transparent;
            display: flex;
            justify-content: center;
            align-items: center;
            height: {size}px;
            font-family: {FONT_STACK};
            overflow: visible;
        }}

        .metric-toggle,
        .metric-info-toggle {{
            display: none;
        }}

        .metric-card {{
            position: relative;
            width: {size}px;
            height: {size}px;
        }}

        .metric-face {{
            position: absolute;
            inset: 0;
            transition: opacity 0.22s ease, transform 0.22s ease;
        }}

        .metric-front {{
            opacity: 1;
            transform: scale(1);
            cursor: pointer;
        }}

        .metric-back {{
            opacity: 0;
            transform: scale(0.97);
            pointer-events: none;
        }}

        .metric-toggle:checked + .metric-card .metric-front {{
            opacity: 0;
            transform: scale(0.97);
            pointer-events: none;
        }}

        .metric-toggle:checked + .metric-card .metric-back {{
            opacity: 1;
            transform: scale(1);
            pointer-events: auto;
        }}

        .back-click-layer {{
            position: absolute;
            inset: 0;
            z-index: 1;
            cursor: pointer;
        }}

        .profile-plane {{
            position: relative;
            z-index: 2;
            opacity: 1;
            transform: scale(1);
            transition: opacity 0.18s ease, transform 0.18s ease;
            pointer-events: none;
        }}

        .profile-info {{
            position: absolute;
            inset: 0;
            z-index: 2;
            opacity: 0;
            transform: scale(0.97);
            pointer-events: none;
            transition: opacity 0.18s ease, transform 0.18s ease;
        }}

        .metric-info-toggle:checked ~ .profile-plane {{
            opacity: 0;
            transform: scale(0.97);
        }}

        .metric-info-toggle:checked ~ .profile-info {{
            opacity: 1;
            transform: scale(1);
        }}

        .info-btn {{
            position: absolute;
            top: 26px;
            right: 26px;
            z-index: 6;
            width: 22px;
            height: 22px;
            border-radius: 999px;
            border: 1px solid rgba(139,92,246,0.38);
            background: rgba(139,92,246,0.13);
            color: #c4b5fd;
            font-size: 13px;
            font-weight: 800;
            line-height: 20px;
            text-align: center;
            cursor: pointer;
            box-sizing: border-box;
            user-select: none;
        }}

        .info-btn:hover {{
            background: rgba(139,92,246,0.22);
            color: #ffffff;
        }}

        svg {{
            overflow: visible;
            display: block;
        }}

        svg text {{
            font-family: {FONT_STACK};
        }}
    </style>
    </head>

    <body>
        <input type="checkbox" id="{widget_id}" class="metric-toggle">

        <div class="metric-card">

            <!-- FRONT -->
            <label for="{widget_id}" class="metric-face metric-front">
                <svg width="{size}" height="{size}" viewBox="-12 -12 244 244" xmlns="http://www.w3.org/2000/svg">
                    <defs>
                        <linearGradient id="{grad_id}" x1="0%" y1="0%" x2="100%" y2="0%">
                            <stop offset="0%"   stop-color="#ea580c"/>
                            <stop offset="18%"  stop-color="#f59e0b"/>
                            <stop offset="36%"  stop-color="#fbbf24"/>
                            <stop offset="46%"  stop-color="#fde68a"/>
                            <stop offset="50%"  stop-color="#f8fafc"/>
                            <stop offset="54%"  stop-color="#ddd6fe"/>
                            <stop offset="64%"  stop-color="#c4b5fd"/>
                            <stop offset="82%"  stop-color="#8b5cf6"/>
                            <stop offset="100%" stop-color="#6d28d9"/>
                        </linearGradient>

                        <filter id="cardGlow" x="-16%" y="-16%" width="132%" height="132%">
                            <feDropShadow dx="0" dy="0"  stdDeviation="0.3" flood-color="rgba(59,130,246,0.12)"/>
                            <feDropShadow dx="0" dy="12" stdDeviation="12"  flood-color="rgba(0,0,0,0.28)"/>
                            <feDropShadow dx="0" dy="0"  stdDeviation="12"  flood-color="rgba(59,130,246,0.08)"/>
                            <feDropShadow dx="0" dy="0"  stdDeviation="22"  flood-color="rgba(139,92,246,0.05)"/>
                        </filter>
                    </defs>

                    <rect x="0" y="0" width="220" height="220" rx="30"
                          fill="{SURFACE_BG}" stroke="rgba(59,130,246,0.18)" stroke-width="1"
                          filter="url(#cardGlow)"/>

                    <text x="110" y="34" text-anchor="middle" font-size="13" font-weight="700" fill="{TEXT_MAIN}">
                        {html.escape(title)}
                    </text>

                    <path d="M 55 150 A 60 60 0 1 1 165 150"
                          fill="none" stroke="{TRACK_BG}" stroke-width="18" stroke-linecap="round"/>

                    <path id="activeArc_{widget_id}" d="M 55 150 A 60 60 0 1 1 165 150"
                          fill="none" stroke="url(#{grad_id})" stroke-width="18"
                          stroke-linecap="round" pathLength="100"/>

                    <circle id="marker_{widget_id}" cx="55" cy="150" r="8.8"
                            fill="{marker_fill}" stroke="{SURFACE_BG}" stroke-width="3"/>

                    <text x="110" y="126" text-anchor="middle" font-size="36" font-weight="800" fill="{marker_fill}">
                        {value_text}
                    </text>

                    <text x="20" y="188" text-anchor="start" font-size="9" font-weight="700" fill="#f59e0b">
                        {html.escape(left_label)}
                    </text>

                    <text x="200" y="188" text-anchor="end" font-size="9" font-weight="700" fill="#8b5cf6">
                        {html.escape(right_label)}
                    </text>
                </svg>
            </label>

            <!-- BACK -->
            <div class="metric-face metric-back">
                <label for="{widget_id}" class="back-click-layer"></label>

                <input type="checkbox" id="{info_id}" class="metric-info-toggle">
                <label for="{info_id}" class="info-btn">i</label>

                <svg class="profile-plane" width="{size}" height="{size}" viewBox="-12 -12 244 244" xmlns="http://www.w3.org/2000/svg">
                    <rect x="0" y="0" width="220" height="220" rx="30"
                          fill="{SURFACE_BG}" stroke="rgba(59,130,246,0.18)" stroke-width="1"/>

                    <text x="110" y="24" text-anchor="middle" font-size="11" font-weight="700" fill="{TEXT_MAIN}">
                        Пространство профилей
                    </text>

                    <rect x="{PLOT_X0}" y="{PLOT_Y0}" width="{PLOT_W/2}" height="{PLOT_H/2}" fill="rgba(139,92,246,0.22)"/>
                    <rect x="{PLOT_MID_X}" y="{PLOT_Y0}" width="{PLOT_W/2}" height="{PLOT_H/2}" fill="rgba(56,189,248,0.18)"/>
                    <rect x="{PLOT_X0}" y="{PLOT_MID_Y}" width="{PLOT_W/2}" height="{PLOT_H/2}" fill="rgba(100,116,139,0.10)"/>
                    <rect x="{PLOT_MID_X}" y="{PLOT_MID_Y}" width="{PLOT_W/2}" height="{PLOT_H/2}" fill="rgba(249,115,22,0.20)"/>

                    <rect x="{PLOT_X0}" y="{PLOT_Y0}" width="{PLOT_W}" height="{PLOT_H}"
                          fill="none" stroke="rgba(255,255,255,0.1)" stroke-width="0.5"/>

                    <line x1="{PLOT_MID_X}" y1="{PLOT_Y0}" x2="{PLOT_MID_X}" y2="{PLOT_Y1}"
                          stroke="rgba(255,255,255,0.14)" stroke-width="0.5"/>

                    <line x1="{PLOT_X0}" y1="{PLOT_MID_Y}" x2="{PLOT_X1}" y2="{PLOT_MID_Y}"
                          stroke="rgba(255,255,255,0.14)" stroke-width="0.5"/>

                    <line x1="{PLOT_X0}" y1="{PLOT_Y1}" x2="{PLOT_X1}" y2="{PLOT_Y0}"
                          stroke="rgba(255,255,255,0.20)" stroke-width="0.8" stroke-dasharray="3,3"/>

                    <text x="{PLOT_X0 + PLOT_W/4:.0f}" y="{PLOT_Y0 + 14:.0f}"
                          text-anchor="middle" font-size="7.5" fill="rgba(148,163,184,0.85)">лекция-монолог</text>

                    <text x="{PLOT_X0 + PLOT_W*3/4:.0f}" y="{PLOT_Y0 + 14:.0f}"
                          text-anchor="middle" font-size="7.5" fill="rgba(148,163,184,0.85)">полный курс</text>

                    <text x="{PLOT_X0 + PLOT_W/4:.0f}" y="{PLOT_Y1 - 5:.0f}"
                          text-anchor="middle" font-size="7.5" fill="rgba(148,163,184,0.85)">поверхностное</text>

                    <text x="{PLOT_X0 + PLOT_W*3/4:.0f}" y="{PLOT_Y1 - 5:.0f}"
                          text-anchor="middle" font-size="7.5" fill="rgba(148,163,184,0.85)">туториал</text>

                    <text x="110" y="206" text-anchor="middle" font-size="8" fill="{TEXT_MUTED}">Инструментальность →</text>

                    <text x="14" y="114" text-anchor="middle" font-size="8" fill="{TEXT_MUTED}"
                          transform="rotate(-90,14,114)">Академичность →</text>

                    <circle cx="{dot_x:.1f}" cy="{dot_y:.1f}" r="7"
                            fill="#F5A623" stroke="{SURFACE_BG}" stroke-width="2.5"/>
                </svg>

                <div class="profile-info">
                    <svg width="{size}" height="{size}" viewBox="-12 -12 244 244" xmlns="http://www.w3.org/2000/svg">
                        <rect x="0" y="0" width="220" height="220" rx="30"
                              fill="{SURFACE_BG}" stroke="rgba(59,130,246,0.18)" stroke-width="1"/>

                        <text x="110" y="34" text-anchor="middle" font-size="12" font-weight="700" fill="{TEXT_MAIN}">
                            Что показывает профиль
                        </text>
                        <foreignObject x="24" y="46" width="172" height="128">
                            <div xmlns="http://www.w3.org/1999/xhtml" style="
                                color:{TEXT_MUTED};
                                font-family:{FONT_STACK};
                                font-size:10.7px;
                                line-height:1.35;
                                text-align:left;
                            ">
                                <b style="color:#c4b5fd;">Академичность</b> —
                                стремление объяснить, почему и как
                                устроен предмет изучения, через
                                теорию и обобщения.

                                <br/>

                                <b style="color:#38bdf8;">Инструментальность</b> —
                                стремление показать, что и как делать:
                                алгоритмы, шаги, техники и способы
                                решения практических задач.
                            </div>
                        </foreignObject>

                        <text x="110" y="194" text-anchor="middle" font-size="8" fill="{TEXT_SOFT}">
                            i — вернуться к плоскости
                        </text>
                    </svg>
                </div>
            </div>
        </div>

        <script>
            const arc = document.getElementById("activeArc_{widget_id}");
            const marker = document.getElementById("marker_{widget_id}");
            const total = arc.getTotalLength();
            const point = arc.getPointAtLength(total * {marker_offset} / 100.0);
            marker.setAttribute("cx", point.x);
            marker.setAttribute("cy", point.y);
        </script>
    </body>
    </html>
    """
    render_html_block(html_block, height=size, width=size)


def render_quality_watch_widget(
    value: int,
    title: str = "Техническое качество",
    subtitle: str = "звук · видео",
    size: int = 280,
):
    value = max(0, min(10, value))
    marker_offset = value * 10

    grad_id = f"grad_quality_{value}_{random.randint(1000,9999)}"
    widget_id = f"quality_{random.randint(100000,999999)}"

    def hex_to_rgb(h: str):
        h = h.lstrip("#")
        return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

    def rgb_to_hex(rgb):
        return "#{:02x}{:02x}{:02x}".format(*rgb)

    def lerp(a, b, t: float):
        return int(round(a + (b - a) * t))

    def blend(c1: str, c2: str, t: float):
        r1, g1, b1 = hex_to_rgb(c1)
        r2, g2, b2 = hex_to_rgb(c2)
        return rgb_to_hex((
            lerp(r1, r2, t),
            lerp(g1, g2, t),
            lerp(b1, b2, t),
        ))

    t = value / 10.0
    if t <= 0.20:
        marker_fill = blend("#ef4444", "#f97316", t / 0.20)
    elif t <= 0.45:
        marker_fill = blend("#f97316", "#f59e0b", (t - 0.20) / (0.45 - 0.20))
    elif t <= 0.70:
        marker_fill = blend("#f59e0b", "#84cc16", (t - 0.45) / (0.70 - 0.45))
    else:
        marker_fill = blend("#84cc16", "#22c55e", (t - 0.70) / (1.00 - 0.70))

    description = (
        "Оценивает, насколько комфортно смотреть и слушать видео: "
        "качество звука, чёткость изображения, стабильность кадра, "
        "читаемость доски или экрана и отсутствие технических помех."
    )

    html_block = f"""
    <!DOCTYPE html>
    <html>
    <head>
    <meta charset="utf-8">
    <style>
        html, body {{
            margin: 0;
            background: transparent;
            display: flex;
            justify-content: center;
            align-items: center;
            height: {size}px;
            font-family: {FONT_STACK};
        }}

        .metric-toggle {{
            display: none;
        }}

        .metric-card {{
            position: relative;
            width: {size}px;
            height: {size}px;
            cursor: pointer;
        }}

        .metric-face {{
            position: absolute;
            inset: 0;
            transition: opacity 0.22s ease, transform 0.22s ease;
        }}

        .metric-front {{
            opacity: 1;
            transform: scale(1);
        }}

        .metric-back {{
            opacity: 0;
            transform: scale(0.97);
            box-sizing: border-box;
            padding: 0;
        }}

        .metric-toggle:checked + .metric-card .metric-front {{
            opacity: 0;
            transform: scale(0.97);
            pointer-events: none;
        }}

        .metric-toggle:checked + .metric-card .metric-back {{
            opacity: 1;
            transform: scale(1);
        }}

        .metric-back-shell {{
            width: 100%;
            height: 100%;
            border-radius: 30px;
            background: {SURFACE_BG};
            border: 1px solid rgba(59, 130, 246, 0.18);
            box-sizing: border-box;
            padding: 22px 18px 18px 18px;
            box-shadow:
                0 0 0 1px rgba(59, 130, 246, 0.12),
                0 12px 12px rgba(0, 0, 0, 0.28),
                0 0 12px rgba(59, 130, 246, 0.08),
                0 0 22px rgba(139, 92, 246, 0.05);
        }}

        .metric-back-title {{
            color: {TEXT_MAIN};
            font-size: 1rem;
            font-weight: 700;
            line-height: 1.2;
            margin-bottom: 12px;
        }}

        .metric-back-desc {{
            color: {TEXT_MUTED};
            font-size: 0.92rem;
            line-height: 1.45;
        }}

        svg text {{
            font-family: {FONT_STACK};
        }}
    </style>
    </head>
    <body>
        <input type="checkbox" id="{widget_id}" class="metric-toggle">
        <label for="{widget_id}" class="metric-card">
            <div class="metric-face metric-front">
                <svg width="{size}" height="{size}" viewBox="-12 -12 244 244" xmlns="http://www.w3.org/2000/svg">
                    <defs>
                        <linearGradient id="{grad_id}" x1="0%" y1="0%" x2="100%" y2="0%">
                            <stop offset="0%" stop-color="#ef4444"/>
                            <stop offset="20%" stop-color="#f97316"/>
                            <stop offset="45%" stop-color="#f59e0b"/>
                            <stop offset="70%" stop-color="#84cc16"/>
                            <stop offset="100%" stop-color="#22c55e"/>
                        </linearGradient>

                        <filter id="cardGlow" x="-16%" y="-16%" width="132%" height="132%" filterUnits="objectBoundingBox">
                            <feDropShadow dx="0" dy="0" stdDeviation="0.3" flood-color="rgba(59, 130, 246, 0.12)"/>
                            <feDropShadow dx="0" dy="12" stdDeviation="12" flood-color="rgba(0, 0, 0, 0.28)"/>
                            <feDropShadow dx="0" dy="0" stdDeviation="12" flood-color="rgba(59, 130, 246, 0.08)"/>
                            <feDropShadow dx="0" dy="0" stdDeviation="22" flood-color="rgba(139, 92, 246, 0.05)"/>
                        </filter>
                    </defs>

                    <rect 
                        x="0" 
                        y="0" 
                        width="220" 
                        height="220" 
                        rx="30" 
                        fill="{SURFACE_BG}" 
                        stroke="rgba(59, 130, 246, 0.18)"
                        stroke-width="1"
                        filter="url(#cardGlow)"
                    />

                    <text x="110" y="34" text-anchor="middle" font-size="13" font-weight="700" fill="{TEXT_MAIN}">
                        {html.escape(title)}
                    </text>

                    <path
                        d="M 55 150 A 60 60 0 1 1 165 150"
                        fill="none"
                        stroke="{TRACK_BG}"
                        stroke-width="18"
                        stroke-linecap="round"
                        pathLength="100"
                    />

                    <path
                        id="activeArcQ"
                        d="M 55 150 A 60 60 0 1 1 165 150"
                        fill="none"
                        stroke="url(#{grad_id})"
                        stroke-width="18"
                        stroke-linecap="round"
                        pathLength="100"
                        stroke-dasharray="{value * 10} 100"
                    />

                    <circle
                        id="markerQ"
                        cx="55"
                        cy="150"
                        r="8.8"
                        fill="{marker_fill}"
                        stroke="{SURFACE_BG}"
                        stroke-width="3"
                    />

                    <text x="110" y="126" text-anchor="middle" font-size="34" font-weight="800" fill="{marker_fill}">
                        {value}
                    </text>

                    <text x="110" y="172" text-anchor="middle" font-size="10.5" fill="{TEXT_MUTED}">
                        {html.escape(subtitle)}
                    </text>
                </svg>
            </div>

            <div class="metric-face metric-back">
                <div class="metric-back-shell">
                    <div class="metric-back-title">{html.escape(title)}</div>
                    <div class="metric-back-desc">{html.escape(description)}</div>
                </div>
            </div>
        </label>

        <script>
            const arc = document.getElementById("activeArcQ");
            const marker = document.getElementById("markerQ");

            const total = arc.getTotalLength();
            const point = arc.getPointAtLength(total * {marker_offset} / 100.0);

            marker.setAttribute("cx", point.x);
            marker.setAttribute("cy", point.y);
        </script>
    </body>
    </html>
    """
    render_html_block(html_block, height=size, width=size)


def render_match_bar(
    value: int,
    title: str = "Соответствие запросу",
    subtitle: str = "интегральная метрика соответствия",
):
    value = max(0, min(100, value))

    bar_html = f"""
    <div style="
        background:{SURFACE_BG};
        border:1px solid rgba(59, 130, 246, 0.18);
        border-radius:24px;
        padding:18px 20px 16px 20px;
        margin-top:8px;
        margin-bottom:10px;
        color:{TEXT_MAIN};
        font-family:{FONT_STACK};
        box-sizing:border-box;
        width:100%;
    box-shadow:
        0 0 0 1px rgba(59, 130, 246, 0.18),
        0 12px 24px rgba(0, 0, 0, 0.28),
        0 0 12px rgba(59, 130, 246, 0.08),
        0 0 22px rgba(139, 92, 246, 0.05);
    ">
        <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:12px;">
            <div style="max-width:75%;">
                <div style="font-size:1.15rem;font-weight:700;color:{TEXT_MAIN};line-height:1.2;">
                    {html.escape(title)}
                </div>
                <div style="font-size:0.92rem;color:{TEXT_MUTED};margin-top:4px;line-height:1.35;">
                    {html.escape(subtitle)}
                </div>
            </div>
            <div style="font-size:2rem;font-weight:800;color:#ffffff;margin-left:16px;">
                {value}%
            </div>
        </div>

        <div style="
            width:100%;
            height:18px;
            background:{TRACK_BG_SOFT};
            border-radius:999px;
            overflow:hidden;
        ">
            <div style="
                width:{value}%;
                height:100%;
                background:linear-gradient(90deg, {ACCENT_CYAN} 0%, {ACCENT_BLUE} 45%, {ACCENT_VIOLET} 100%);
                border-radius:999px;
            "></div>
        </div>

        <div style="
            display:flex;
            justify-content:space-between;
            margin-top:8px;
            font-size:0.82rem;
            color:{TEXT_MUTED};
        ">
            <span>Слабое совпадение</span>
            <span>Частичное</span>
            <span>Высокое совпадение</span>
        </div>
    </div>
    """
    render_html_block(bar_html, height=185, width=None)


def render_role_instruction(user_type: str):
    """Карточка-инструкция зависящая от роли. Вставить ПЕРЕД st.video()."""
    instr = ROLE_INSTRUCTIONS.get(user_type)
    if not instr:
        return
 
    html_block = f"""
    <div style="
        background: {SURFACE_BG};
        border: 1px solid {BORDER};
        border-radius: 20px;
        padding: 18px 22px 16px 22px;
        margin-bottom: 16px;
        font-family: {FONT_STACK};
        box-shadow:
            0 0 0 1px rgba(59,130,246,0.12),
            0 8px 24px rgba(0,0,0,0.22);
    ">
        <div style="
            display:flex;
            justify-content:space-between;
            align-items:center;
            margin-bottom:10px;
        ">
            <div style="
                font-size: 0.78rem;
                font-weight: 700;
                text-transform: uppercase;
                letter-spacing: 0.08em;
                color: {TEXT_SOFT};
            ">
                Как читать этот отчёт
            </div>

            <div style="
                padding: 4px 10px;
                border-radius: 999px;
                background: rgba(139,92,246,0.12);
                border: 1px solid rgba(139,92,246,0.35);
                color: {ACCENT_VIOLET};
                font-size: 0.72rem;
                font-weight: 700;
                white-space: nowrap;
            ">
                {ROLE_ICONS.get(user_type, "👤")} Выбрана роль: {html.escape(user_type)}
            </div>
        </div>
 
        <div style="
            display: flex;
            gap: 20px;
            flex-wrap: wrap;
        ">
            <div style="flex: 1; min-width: 200px;">
                <div style="font-size: 0.78rem; font-weight: 700; color: {ACCENT_CYAN}; margin-bottom: 4px;">
                    С чего начать
                </div>
                <div style="font-size: 0.87rem; color: {TEXT_MUTED}; line-height: 1.5;">
                    {html.escape(instr['start'])}
                </div>
            </div>
            <div style="flex: 2; min-width: 260px;">
                <div style="font-size: 0.78rem; font-weight: 700; color: {ACCENT_VIOLET}; margin-bottom: 4px;">
                    Как читать
                </div>
                <div style="font-size: 0.87rem; color: {TEXT_MUTED}; line-height: 1.5;">
                    {html.escape(instr['how'])}
                </div>
            </div>
        </div>
    </div>
    """
    role_instruction_heights = {
        "Обучающийся": 165,
        "Спикер": 185,
        "Продакшн": 205,
        "Заказчик": 185,
    }

    render_html_block(
        html_block,
        height=role_instruction_heights.get(user_type, 205),
    )
 
 
def _fmt_timecode(seconds: float) -> str:
    seconds = int(seconds)
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    if h:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"
 
 
def render_narrative_and_timecodes(result: dict):
    """
    Блок: нарратив во всю ширину + двухколоночная секция (таймкоды 70% / доп инфо 30%).
    Вставить ПОСЛЕ render_match_bar() в render_result_panel().
    """
    narrative_data = result.get("narrative", {})
    narrative_text = narrative_data.get("narrative", "")
    segments       = narrative_data.get("segments", [])
    prerequisites  = narrative_data.get("prerequisites", [])
    gaps           = narrative_data.get("topic_coverage", {}).get("gaps", [])
    covered        = narrative_data.get("topic_coverage", {}).get("covered", [])
    info_density   = narrative_data.get("info_density", "")
    learning_path  = narrative_data.get("learning_path", {})
    title_match    = narrative_data.get("title_match", {})
    video_url      = result.get("video_url", "")
 
    # ── Нарратив во всю ширину ───────────────────────────────────────────────
    if narrative_text:
        narrative_html = f"""
        <div style="
            background: {SURFACE_BG};
            border: 1px solid {BORDER};
            border-radius: 20px;
            padding: 20px 24px;
            margin-bottom: 18px;
            font-family: {FONT_STACK};
            box-shadow:
                0 0 0 1px rgba(59,130,246,0.10),
                0 8px 20px rgba(0,0,0,0.20);
        ">
            <div style="
                font-size: 0.78rem;
                font-weight: 700;
                text-transform: uppercase;
                letter-spacing: 0.08em;
                color: {TEXT_SOFT};
                margin-bottom: 10px;
            ">О видео</div>
            <div style="
                font-size: 0.97rem;
                color: {TEXT_MAIN};
                line-height: 1.65;
            ">{html.escape(narrative_text)}</div>
        </div>
        """
    narrative_height = 90 + math.ceil(len(narrative_text) / 110) * 22
    render_html_block(narrative_html, height=narrative_height)
 
    # ── Двухколоночная секция ────────────────────────────────────────────────
    left_col, right_col = st.columns([6.5, 3.5], gap="small")
 
    # ── Левая колонка: таймкоды ──────────────────────────────────────────────
    with left_col:
        if segments:
            # Строим YouTube deep-link: ?t=секунды
            def yt_link(url: str, start: float) -> str:
                if not url:
                    return "#"
                base = url.split("&t=")[0].split("?t=")[0]
                sep = "&" if "?" in base else "?"
                return f"{base}{sep}t={int(start)}"
 
            items_html = ""
            for seg in segments:
                title       = html.escape(seg.get("title", ""))
                description = html.escape(seg.get("description", ""))
                start       = seg.get("start", 0)
                tc          = _fmt_timecode(start)
                link        = yt_link(video_url, start)
 
                items_html += f"""
                <div style="
                    display: flex;
                    gap: 14px;
                    padding: 12px 0;
                    border-bottom: 1px solid {BORDER};
                    align-items: flex-start;
                ">
                    <a href="{link}" target="_blank" style="
                        flex-shrink: 0;
                        display: inline-block;
                        padding: 3px 10px;
                        background: {SURFACE_BG_ELEVATED};
                        border: 1px solid {BORDER};
                        border-radius: 8px;
                        font-size: 0.78rem;
                        font-weight: 700;
                        color: {ACCENT_CYAN};
                        text-decoration: none;
                        white-space: nowrap;
                        margin-top: 2px;
                        font-family: 'SF Mono', 'Fira Code', monospace;
                    ">{tc}</a>
                    <div>
                        <div style="
                            font-size: 0.92rem;
                            font-weight: 700;
                            color: {TEXT_MAIN};
                            margin-bottom: 3px;
                            line-height: 1.3;
                        ">{title}</div>
                        <div style="
                            font-size: 0.82rem;
                            color: {TEXT_MUTED};
                            line-height: 1.45;
                        ">{description}</div>
                    </div>
                </div>
                """
 
            timecodes_html = f"""
            <div style="
                background: {SURFACE_BG};
                border: 1px solid {BORDER};
                border-radius: 20px;
                padding: 16px 20px 8px 20px;
                font-family: {FONT_STACK};
                box-shadow:
                    0 0 0 1px rgba(59,130,246,0.10),
                    0 8px 20px rgba(0,0,0,0.20);
            ">
                <div style="
                    font-size: 0.78rem;
                    font-weight: 700;
                    text-transform: uppercase;
                    letter-spacing: 0.08em;
                    color: {TEXT_SOFT};
                    margin-bottom: 6px;
                ">Темы и таймкоды</div>
                {items_html}
            </div>
            """
            est_height = 70

            for seg in segments:
                title_len = len(seg.get("title", ""))
                desc_len = len(seg.get("description", ""))

                title_lines = max(1, math.ceil(title_len / 58))
                desc_lines = max(1, math.ceil(desc_len / 90))

                est_height += 26 + title_lines * 25 + desc_lines * 16 + 10

            est_height += 60
            render_html_block(timecodes_html, height=est_height)
 
    # ── Правая колонка: доп инфо ─────────────────────────────────────────────
    with right_col:
        density_labels = {
            "low":    ("Низкая",    TEXT_MUTED),
            "medium": ("Средняя",   ACCENT_BLUE),
            "high":   ("Высокая",   ACCENT_VIOLET),
        }
        density_text, density_color = density_labels.get(
            info_density, ("—", TEXT_SOFT)
        )
 
        path_labels = {
            "intro":          ("Вводная лекция",          ACCENT_CYAN),
            "part_of_course": ("Часть курса",             ACCENT_BLUE),
            "standalone":     ("Самостоятельная тема",    ACCENT_GREEN),
        }
        path_text, path_color = path_labels.get(
            learning_path.get("type", ""), ("—", TEXT_SOFT)
        )
 
        match_labels = {
            "full":    ("Точное",    ACCENT_GREEN),
            "partial": ("Частичное", ACCENT_BLUE),
            "none":    ("Не совпадает", "#ef4444"),
        }
        match_text, match_color = match_labels.get(
            title_match.get("verdict", ""), ("—", TEXT_SOFT)
        )
 
        # Пресреквизиты
        prereqs_html = ""
        if prerequisites:
            conf_colors = {
                "high":   ACCENT_GREEN,
                "medium": ACCENT_BLUE,
                "low":    TEXT_MUTED,
            }
            conf_labels = {"high": "нужно", "medium": "желательно", "low": "полезно"}
            for p in prerequisites:
                concept = html.escape(p.get("concept", ""))
                conf    = p.get("confidence", "medium")
                c_color = conf_colors.get(conf, TEXT_MUTED)
                c_label = conf_labels.get(conf, "")
                prereqs_html += f"""
                <div style="
                    display: flex;
                    justify-content: space-between;
                    align-items: flex-start;
                    gap: 8px;
                    padding: 6px 0;
                    border-bottom: 1px solid {BORDER};
                    font-size: 0.82rem;
                ">
                    <span style="color: {TEXT_MAIN}; line-height: 1.4;">{concept}</span>
                    <span style="
                        flex-shrink: 0;
                        font-size: 0.72rem;
                        font-weight: 600;
                        color: {c_color};
                        padding-top: 2px;
                    ">{c_label}</span>
                </div>
                """
 
        # Пробелы
        gaps_html = ""
        if gaps:
            for g in gaps:
                gaps_html += f"""
                <div style="
                    padding: 5px 0;
                    border-bottom: 1px solid {BORDER};
                    font-size: 0.82rem;
                    color: {TEXT_MUTED};
                ">· {html.escape(g)}</div>
                """
        else:
            gaps_html = f"""
            <div style="font-size:0.82rem; color:{TEXT_SOFT}; padding: 5px 0;">
                Пробелов не выявлено
            </div>
            """
 
        side_html = f"""
        <div style="
            background: {SURFACE_BG};
            border: 1px solid {BORDER};
            border-radius: 20px;
            padding: 16px 18px;
            font-family: {FONT_STACK};
            box-shadow:
                0 0 0 1px rgba(59,130,246,0.10),
                0 8px 20px rgba(0,0,0,0.20);
        ">
            <!-- Быстрые метки -->
            <div style="display:flex; flex-direction:column; gap:8px; margin-bottom:16px;">
 
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <span style="font-size:0.78rem; color:{TEXT_SOFT};">Плотность</span>
                    <span style="font-size:0.82rem; font-weight:700; color:{density_color};">{density_text}</span>
                </div>
 
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <span style="font-size:0.78rem; color:{TEXT_SOFT};">Тип видео</span>
                    <span style="font-size:0.82rem; font-weight:700; color:{path_color};">{path_text}</span>
                </div>
 
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <span style="font-size:0.78rem; color:{TEXT_SOFT};">Название</span>
                    <span style="font-size:0.82rem; font-weight:700; color:{match_color};">{match_text}</span>
                </div>
 
            </div>
 
            <div style="height:1px; background:{BORDER}; margin-bottom:14px;"></div>
 
            <!-- Пресреквизиты -->
            <div style="
                font-size: 0.75rem;
                font-weight: 700;
                text-transform: uppercase;
                letter-spacing: 0.07em;
                color: {TEXT_SOFT};
                margin-bottom: 6px;
            ">Нужно знать заранее</div>
            {prereqs_html}
 
            <div style="height:1px; background:{BORDER}; margin: 14px 0;"></div>
 
            <!-- Пробелы -->
            <div style="
                font-size: 0.75rem;
                font-weight: 700;
                text-transform: uppercase;
                letter-spacing: 0.07em;
                color: {TEXT_SOFT};
                margin-bottom: 6px;
            ">Пробелы в покрытии</div>
            {gaps_html}
 
            <!-- Пояснение к названию -->
            {"" if not title_match.get("explanation") else f'''
            <div style="height:1px; background:{BORDER}; margin: 14px 0;"></div>
            <div style="font-size:0.78rem; color:{TEXT_MUTED}; line-height:1.45;">
                {html.escape(title_match.get("explanation",""))}
            </div>
            '''}
        </div>
        """
 
        est_side_height = 150

        for p in prerequisites:
            concept_len = len(p.get("concept", ""))
            concept_lines = max(1, math.ceil(concept_len / 28))
            est_side_height += 20 + concept_lines * 17

        if gaps:
            for g in gaps:
                gap_lines = max(1, math.ceil(len(g) / 30))
                est_side_height += 14 + gap_lines * 17
        else:
            est_side_height += 34

        if title_match.get("explanation"):
            explanation_len = len(title_match.get("explanation", ""))
            explanation_lines = max(1, math.ceil(explanation_len / 32))
            est_side_height += 32 + explanation_lines * 17

        est_side_height += 28

        render_html_block(side_html, height=est_side_height)


def init_state():
    defaults = {
        "app_state": "idle",
        "analysis_started": False,
        "submitted_data": None,
        "progress_step": 0,
        "form_errors": {},
        "has_result": False,
        "current_result": None,
        "results_history": [],
        "result_counter": 0,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def render_welcome_panel():
    st.markdown("## Добро пожаловать")
    st.write(
        """
        Этот инструмент предназначен для предварительной оценки образовательного видео
        с точки зрения качества подачи, структуры, соответствия аудитории и особенностей восприятия.

        На текущем этапе это прототип интерфейса, который демонстрирует,
        как будет выглядеть взаимодействие пользователя с системой.
        """
    )

    st.markdown("### Что делает инструмент")
    st.write(
        """
        - принимает ссылку на образовательное видео;
        - учитывает, кто использует инструмент и для какой аудитории оценивается материал;
        - позволяет учесть цели просмотра или использования контента;
        - формирует аналитическую выдачу с метриками и рекомендациями.
        """
    )

    st.markdown("### Как пользоваться")
    st.write(
        """
        1. Вставьте ссылку на видео в левом блоке.  
        2. Выберите свою роль.  
        3. Укажите уровень образования.  
        4. Отметьте цели / запросы.  
        5. Нажмите кнопку **«Продолжить»**.
        """
    )

    st.markdown("### Что появится после запуска")
    st.write(
        """
        После отправки формы здесь будут отображаться:
        - этапы анализа,
        - индикатор выполнения,
        - итоговые метрики,
        - текстовое описание,
        - рекомендации.
        """
    )

    st.info(
        "Пока анализ не запущен, справа отображается справочная информация о работе инструмента."
    )


def extract_video_id(url: str) -> str:
    try:
        parsed = urlparse(url.strip())
        domain = parsed.netloc.lower()
        path = parsed.path
        query = parse_qs(parsed.query)

        if domain in {"youtu.be", "www.youtu.be"}:
            return path.strip("/")

        if domain in {"youtube.com", "www.youtube.com", "m.youtube.com"}:
            if path == "/watch" and "v" in query and query["v"]:
                return query["v"][0]
            if path.startswith("/shorts/"):
                parts = path.strip("/").split("/")
                if len(parts) >= 2:
                    return parts[1]
        return ""
    except Exception:
        return ""


def mock_video_title(url: str, result_id: int) -> str:
    video_id = extract_video_id(url)
    if video_id:
        return f"YouTube Video ({video_id})"
    return f"Видео #{result_id}"


def render_tag(text: str, bg: str = SURFACE_BG_ELEVATED, color: str = "#93c5fd") -> str:
    return (
        f'<span style="display:inline-block;padding:4px 10px;margin:2px 6px 2px 0;'
        f'border-radius:999px;background:{bg};color:{color};font-size:0.85rem;'
        f'font-weight:600;font-family:{FONT_STACK};'
        f'box-shadow:0 0 0 1px rgba(30,41,59,0.30), 0 0 14px rgba(59,130,246,0.06);">'
        f'{html.escape(text)}</span>'
    )


def render_metric_bar(label: str, value: float, color: str, display_value: str | None = None) -> str:
    bar_value = max(0, min(100, value))
    shown = display_value if display_value is not None else f"{bar_value:g}"

    return f"""
    <div style="margin: 8px 0 10px 0; font-family:{FONT_STACK};">
        <div style="display:flex;justify-content:space-between;font-size:0.9rem;margin-bottom:4px;color:{TEXT_MAIN};">
            <span>{html.escape(label)}</span>
            <span><b>{shown}</b></span>
        </div>
        <div style="width:100%;height:8px;background:{TRACK_BG};border-radius:999px;overflow:hidden;">
            <div style="width:{bar_value}%;height:100%;background:{color};border-radius:999px;"></div>
        </div>
    </div>
    """


def build_mock_result():
    st.session_state.result_counter += 1
    data = st.session_state.submitted_data or {}

    delivery_profile = random.randint(-100, 100)
    tech_quality = random.randint(45, 95)
    request_match = random.randint(50, 96)

    video_url = data.get("video_url", "")
    result_id = st.session_state.result_counter

    metrics = {
        "delivery_profile": delivery_profile,
        "tech_quality": tech_quality,
        "request_match": request_match,
    }

    result = {
        "id": result_id,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "video_url": video_url,
        "video_title": mock_video_title(video_url, result_id),
        "user_type": data.get("user_type", ""),
        "audience_level": data.get("audience_level", ""),
        "immersion_level": data.get("immersion_level", ""),
        "view_goal": data.get("view_goal", ""),
        "summary": (
            "Это демонстрационная заглушка результата анализа. "
            "Здесь будет храниться краткое текстовое описание видео, "
            "вывод о его структуре, качестве подачи и соответствии выбранной аудитории."
        ),
        "metrics" :{
            "academic_score":     0.75,
            "instrumental_score": 0.10,
            "tech_quality":       5.1,
        },
        "narrative": MOCK_NARRATIVE,
        "warnings": build_warning_flags(metrics),
    }
    return result


def render_running_panel():
    st.markdown("## Выполняется анализ")
    st.write("Система обрабатывает видео и подготавливает результат.")

    progress_bar = st.progress(0, text="Подготовка к запуску")
    status_placeholder = st.empty()

    def update_progress(text: str, value: int):
        value = max(0, min(100, int(value)))
        progress_bar.progress(value, text=text)
        status_placeholder.info(f"⏳ {text}")

    try:
        update_progress("Запуск пайплайна", 1)

        pipeline_output = run_pipeline(
            st.session_state.submitted_data["video_url"],
            user_params=st.session_state.submitted_data,
            progress_callback=update_progress,
        )

        result = build_result_from_pipeline_output(
            pipeline_output,
            st.session_state.submitted_data,
        )

        st.session_state.result_counter += 1
        result["id"] = st.session_state.result_counter

        st.session_state.current_result = result
        st.session_state.results_history.insert(0, result)
        st.session_state.results_history = st.session_state.results_history[:10]

        st.session_state.app_state = "done"
        st.session_state.has_result = True
        st.session_state.progress_step = 0

        update_progress("Готово", 100)
        st.rerun()

    except Exception as e:
        st.session_state.app_state = "idle"
        st.session_state.has_result = False
        st.error(f"Ошибка при анализе видео: {e}")


def render_result_panel(result: dict):
    st.markdown("## Результат анализа")

    submitted_data = st.session_state.submitted_data or {}
    user_type = submitted_data.get("user_type", "Обучающийся")
    render_role_instruction(user_type)

    st.markdown(
        "<div style='height:0px; margin-top:-1034px;'></div>",
        unsafe_allow_html=True,
    )

    video_url = result.get("video_url", "")
    if video_url:
        st.video(video_url)
    else:
        st.info("Превью видео недоступно.")

    metrics = result.get("metrics", {})

    academic_level     = metrics.get("academic_level",     random.randint(1, 5))
    instrumental_level = metrics.get("instrumental_level", random.randint(1, 5))
    _LEVEL_MID = {1: 0.10, 2: 0.30, 3: 0.52, 4: 0.75, 5: 0.92}

    academic_score     = metrics.get("academic_score",     _LEVEL_MID[academic_level])
    instrumental_score = metrics.get("instrumental_score", _LEVEL_MID[instrumental_level])
    tech_quality = metrics.get("tech_quality", 82)
    request_match = metrics.get("request_match", 74)
    warnings = result.get("warnings", build_warning_flags(metrics))

    st.markdown("### Ключевые метрики")

    outer_left, outer_center, outer_right = st.columns([1.08, 0.98, 1.08], gap="medium")

    with outer_left:
        left_content, left_spacer = st.columns([1, 0.08], gap="small")
        with left_content:
            render_bipolar_watch_widget(
                value=round((academic_score - instrumental_score) * 100),
                academic_score=academic_score,
                instrumental_score=instrumental_score,
                size=280,
            )

    with outer_center:
        # center_spacer_left, center_content, center_spacer_right = st.columns([0.04, 0.92, 0.04], gap="small")
        # with center_content:
            render_warning_flags(warnings, columns=2)

    with outer_right:
        right_spacer, right_content = st.columns([0.08, 1], gap="small")
        with right_content:
            render_quality_watch_widget(
                value=tech_quality,
                title="Техническое качество",
                subtitle="звук · видео",
                size=280,
            )

    # render_match_bar(
    #     value=request_match,
    #     title="Соответствие запросу",
    #     subtitle="интегральная метрика соответствия видео выбранному сценарию",
    # )

    render_narrative_and_timecodes(result)


def render_history_panel():
    history = st.session_state.results_history

    if not history:
        return

    with st.expander("История прошлых запросов", expanded=False):
        for item in history:
            metrics = item.get("metrics", {})
            immersion_level = item.get("immersion_level", "")
            view_goal = item.get("view_goal", "")

            st.markdown(f"**#{item['id']} · {item['created_at']}**")

            video_title = item.get("video_title", f"Видео #{item['id']}")
            video_url = item.get("video_url", "")
            if video_url:
                st.markdown(f"[{video_title}]({video_url})")
            else:
                st.markdown(f"**{video_title}**")

            meta_html = (
                render_tag(item["user_type"], bg=SURFACE_BG_ELEVATED, color="#67e8f9")
                + render_tag(item["audience_level"], bg=SURFACE_BG_ELEVATED, color="#86efac")
            )
            st.markdown(meta_html, unsafe_allow_html=True)

            scenario_html = ""
            if immersion_level:
                scenario_html += render_tag(immersion_level, bg=TRACK_BG, color=TEXT_MAIN)
            if view_goal:
                scenario_html += render_tag(view_goal, bg=TRACK_BG, color=TEXT_MAIN)

            if scenario_html:
                st.markdown(scenario_html, unsafe_allow_html=True)

            st.write(item["summary"])

            academic_score = float(metrics.get("academic_score", 0))
            instrumental_score = float(metrics.get("instrumental_score", 0))
            tech_quality = float(metrics.get("tech_quality", 0))

            bars_html = (
                render_metric_bar(
                    "Академичность",
                    academic_score * 100,
                    ACCENT_VIOLET,
                    display_value=f"{academic_score * 100:.0f}",
                )
                + render_metric_bar(
                    "Инструментальность",
                    instrumental_score * 100,
                    ACCENT_CYAN,
                    display_value=f"{instrumental_score * 100:.0f}",
                )
                + render_metric_bar(
                    "Тех. качество",
                    tech_quality * 10,
                    ACCENT_GREEN,
                    display_value=f"{tech_quality:g}",
                )
            )
            render_html_block(bars_html, height=170)

            _, btn_col = st.columns([5, 1])
            with btn_col:
                if st.button("Открыть", key=f"open_result_{item['id']}"):
                    st.session_state.current_result = item
                    st.session_state.has_result = True
                    st.session_state.app_state = "done"
                    st.rerun()

            st.divider()


def is_youtube_url(url: str) -> bool:
    try:
        parsed = urlparse(url.strip())
        domain = parsed.netloc.lower()

        youtube_domains = {
            "youtube.com",
            "www.youtube.com",
            "m.youtube.com",
            "youtu.be",
            "www.youtu.be",
        }
        return domain in youtube_domains
    except Exception:
        return False


def looks_like_youtube_video_url(url: str) -> bool:
    try:
        parsed = urlparse(url.strip())
        domain = parsed.netloc.lower()
        path = parsed.path
        query = parse_qs(parsed.query)

        if domain in {"youtu.be", "www.youtu.be"}:
            return len(path.strip("/")) > 0

        if domain in {"youtube.com", "www.youtube.com", "m.youtube.com"}:
            if path == "/watch" and "v" in query and query["v"]:
                return True
            if path.startswith("/shorts/") and len(path.split("/")) > 2:
                return True

        return False
    except Exception:
        return False


def validate_form(video_url: str, immersion_level: str, view_goal: str) -> dict:
    errors = {}
    cleaned_url = video_url.strip()

    if not cleaned_url:
        errors["video_url"] = "Добавьте ссылку на видео."
    elif not is_youtube_url(cleaned_url):
        errors["video_url"] = "Сейчас поддерживаются только ссылки на YouTube."
    elif not looks_like_youtube_video_url(cleaned_url):
        errors["video_url"] = (
            "Укажите ссылку на конкретное видео, а не на главную страницу "
            "или другой раздел YouTube."
        )

    if not immersion_level:
        errors["immersion_level"] = "Выберите уровень погружения в тему."

    if not view_goal:
        errors["view_goal"] = "Выберите цель просмотра."

    return errors


init_state()
inject_global_styles()

st.title("Анализ образовательного видео")

left_col, right_col = st.columns([1, 2.5], gap="large")

with left_col:

    if st.button("Демо-режим: готовые результаты", width="stretch", type="primary", icon="🚀"):
        demo_results = load_demo_results()

        st.session_state.demo_user_type = "Обучающийся"
        st.session_state.demo_audience_level = "Бакалавр (1-2 курс)"
        st.session_state.demo_immersion_level = "Знаю частично"
        st.session_state.demo_view_goal = "Составить общее представление"

        st.session_state.results_history = []
        st.session_state.current_result = None
        st.session_state.result_counter = 0

        st.session_state.submitted_data = {
            "video_url": demo_results[0]["video_url"],
            "user_type": "Обучающийся",
            "audience_level": "Бакалавр (1-2 курс)",
            "immersion_level": "Знаю частично",
            "view_goal": "Составить общее представление",
        }
        st.session_state.current_result = demo_results[0]
        st.session_state.results_history = demo_results
        st.session_state.result_counter = len(demo_results)
        st.session_state.has_result = True
        st.session_state.app_state = "done"
        st.rerun()

    with st.form("video_input_form"):
        
        st.markdown("### Ссылка на видео")

        if "video_url" in st.session_state.form_errors:
            st.warning(f"❗ {st.session_state.form_errors['video_url']}")

        video_url = st.text_input(
            "URL видео",
            placeholder="https://www.youtube.com/watch?v=...",
            label_visibility="collapsed",
        )

        st.divider()

        st.markdown("### Кто я?")
        user_type = st.radio(
            "Тип пользователя",
            options=USER_TYPES,
            label_visibility="collapsed",
            key="demo_user_type",
        )

        st.divider()

        st.markdown("### Уровень образования")
        st.caption(
            "💡 Если Вы обучающийся — выберите свой уровень. "
            "В остальных случаях — уровень целевой аудитории."
        )

        audience_level = st.radio(
            "Уровень образования",
            options=LEVELS,
            label_visibility="collapsed",
            key="demo_audience_level",
        )

        st.divider()

        st.markdown("### Уровень погружённости")
        if "immersion_level" in st.session_state.form_errors:
            st.warning(f"❗ {st.session_state.form_errors['immersion_level']}")

        immersion_level = st.radio(
            "Уровень погружённости",
            options=IMMERSION_LEVELS,
            label_visibility="collapsed",
            key="demo_immersion_level",
        )

        st.divider()

        st.markdown("### Цель просмотра")
        if "view_goal" in st.session_state.form_errors:
            st.warning(f"❗ {st.session_state.form_errors['view_goal']}")

        view_goal = st.radio(
            "Цель просмотра",
            options=VIEW_GOALS,
            label_visibility="collapsed",
            key="demo_view_goal",
        )

        st.divider()

        if st.session_state.form_errors:
            st.error(
                "Пожалуйста, исправьте ошибки:\n\n"
                + "\n".join([f"- {msg}" for msg in st.session_state.form_errors.values()])
            )

        submitted = st.form_submit_button("Продолжить", width="stretch")


if submitted:
    errors = validate_form(video_url, immersion_level, view_goal)

    if errors:
        st.session_state.form_errors = errors
        st.rerun()
    else:
        st.session_state.form_errors = {}

        st.session_state.results_history = []
        st.session_state.current_result = None
        st.session_state.result_counter = 0

        st.session_state.submitted_data = {
            "video_url": video_url.strip(),
            "user_type": user_type,
            "audience_level": audience_level,
            "immersion_level": immersion_level,
            "view_goal": view_goal,
        }

        st.session_state.app_state = "running"
        st.session_state.progress_step = 0
        st.rerun()


with right_col:
    right_placeholder = st.empty()

    state = st.session_state.app_state
    has_result = st.session_state.has_result

    with right_placeholder.container():
        if state == "running":
            render_running_panel()

        elif state == "done" or has_result:
            current_result = st.session_state.current_result
            if current_result:
                render_result_panel(current_result)
                render_history_panel()
            else:
                render_welcome_panel()

        else:
            render_welcome_panel()

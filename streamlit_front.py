import time
import random
import math
import html
from urllib.parse import urlparse, parse_qs

import streamlit as st
import streamlit.components.v1 as components


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

USER_TYPES = [
    "Обучающийся",
    "Автор контента",
    "Образовательная организация",
    "Заказчик контента",
]

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

WARNING_ICON_CHAOS = """
<svg width="220" height="180" viewBox="0 0 220 180" xmlns="http://www.w3.org/2000/svg">
  <g fill="none" stroke="currentColor" stroke-width="4.5" stroke-linecap="round" stroke-linejoin="round">
    <circle cx="18" cy="88" r="4.5" fill="currentColor" stroke="none"/>
    <path d="
      M 30 88
      C 52 88, 56 78, 68 60
      C 78 44, 98 48, 98 70
      C 98 94, 78 108, 58 120
      C 34 134, 34 154, 58 156
      C 82 158, 92 138, 82 116
      C 72 94, 56 72, 44 56
      C 34 42, 34 26, 50 22
      C 68 18, 82 34, 80 54
      C 78 76, 58 88, 52 108
      C 46 128, 58 146, 82 150
      C 108 154, 122 140, 128 118
      C 134 96, 122 78, 104 72
      C 86 66, 74 78, 74 96
      C 74 116, 88 132, 112 136
      C 140 140, 160 126, 160 100
      C 160 74, 142 60, 120 64
      C 98 68, 92 88, 102 104
      C 114 122, 144 126, 166 126
      C 182 126, 192 122, 200 112
    "/>
    <path d="M 188 100 L 202 114 L 188 128"/>
  </g>
</svg>
"""

WARNING_ICON_VISUAL_NOISE = """
<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none"
stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
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
<svg width="160" height="160" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
  <g fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
    <circle cx="12" cy="12" r="9"/>
    <path d="M12 8v4"/>
    <circle cx="12" cy="16" r="0.8" fill="currentColor" stroke="none"/>
  </g>
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
            "title": "Хаотичность",
            "description": "Перескакивание между мыслями, слабая связность и хаотичные переходы в объяснении.",
            "active": random.random() < 0.4,
            "icon_svg": WARNING_ICON_CHAOS,
        },
        {
            "title": "Визуальный шум",
            "description": "Лишние движения, отвлекающие элементы в кадре или перегруженная визуальная сцена.",
            "active": random.random() < 0.3,
            "icon_svg": WARNING_ICON_VISUAL_NOISE,
        },
        {
            "title": "Унылость",
            "description": "Монотонность, слабая динамика речи и утомляющая однородность подачи.",
            "active": random.random() < 0.35,
            "icon_svg": WARNING_ICON_MONOTONY,
        },
        {
            "title": "Эмоциональность",
            "description": "Временная заглушка под будущий warning-сигнал.",
            "active": random.random() < 0.2,
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
    left_label: str = "Инструментальность",
    right_label: str = "Академичность",
    size: int = 280,
):
    value = max(-100, min(100, value))
    normalized = (value + 100) / 200
    marker_offset = normalized * 100

    grad_id = f"grad_bipolar_{abs(value)}_{random.randint(1000,9999)}"
    widget_id = f"bipolar_{random.randint(100000,999999)}"

    value_text = str(abs(value))

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

    t = normalized
    if t <= 0.18:
        marker_fill = blend("#ea580c", "#f59e0b", t / 0.18)
    elif t <= 0.36:
        marker_fill = blend("#f59e0b", "#fbbf24", (t - 0.18) / (0.36 - 0.18))
    elif t <= 0.46:
        marker_fill = blend("#fbbf24", "#fde68a", (t - 0.36) / (0.46 - 0.36))
    elif t <= 0.50:
        marker_fill = blend("#fde68a", "#f8fafc", (t - 0.46) / (0.50 - 0.46))
    elif t <= 0.54:
        marker_fill = blend("#f8fafc", "#ddd6fe", (t - 0.50) / (0.54 - 0.50))
    elif t <= 0.64:
        marker_fill = blend("#ddd6fe", "#c4b5fd", (t - 0.54) / (0.64 - 0.54))
    elif t <= 0.82:
        marker_fill = blend("#c4b5fd", "#8b5cf6", (t - 0.64) / (0.82 - 0.64))
    else:
        marker_fill = blend("#8b5cf6", "#6d28d9", (t - 0.82) / (1.00 - 0.82))

    description = (
        "Заглушка описания метрики. Здесь будет пояснение, что означает профиль подачи, "
        "как интерпретировать текущее положение на шкале и почему значение смещено в сторону "
        "более инструментальной или более академической подачи."
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
            overflow: visible;
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
        <label for="{widget_id}" class="metric-card">
            <div class="metric-face metric-front">
                <svg width="{size}" height="{size}" viewBox="-12 -12 244 244" xmlns="http://www.w3.org/2000/svg">
                    <defs>
                        <linearGradient id="{grad_id}" x1="0%" y1="0%" x2="100%" y2="0%">
                            <stop offset="0%" stop-color="#ea580c"/>
                            <stop offset="18%" stop-color="#f59e0b"/>
                            <stop offset="36%" stop-color="#fbbf24"/>
                            <stop offset="46%" stop-color="#fde68a"/>
                            <stop offset="50%" stop-color="#f8fafc"/>
                            <stop offset="54%" stop-color="#ddd6fe"/>
                            <stop offset="64%" stop-color="#c4b5fd"/>
                            <stop offset="82%" stop-color="#8b5cf6"/>
                            <stop offset="100%" stop-color="#6d28d9"/>
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
                    />

                    <path
                        id="activeArc"
                        d="M 55 150 A 60 60 0 1 1 165 150"
                        fill="none"
                        stroke="url(#{grad_id})"
                        stroke-width="18"
                        stroke-linecap="round"
                        pathLength="100"
                    />

                    <circle
                        id="marker"
                        cx="55"
                        cy="150"
                        r="8.8"
                        fill="{marker_fill}"
                        stroke="{SURFACE_BG}"
                        stroke-width="3"
                    />

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
            </div>

            <div class="metric-face metric-back">
                <div class="metric-back-shell">
                    <div class="metric-back-title">{html.escape(title)}</div>
                    <div class="metric-back-desc">{html.escape(description)}</div>
                </div>
            </div>
        </label>

        <script>
            const arc = document.getElementById("activeArc");
            const marker = document.getElementById("marker");

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
    value = max(0, min(100, value))
    marker_offset = value

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

    t = value / 100.0
    if t <= 0.20:
        marker_fill = blend("#ef4444", "#f97316", t / 0.20)
    elif t <= 0.45:
        marker_fill = blend("#f97316", "#f59e0b", (t - 0.20) / (0.45 - 0.20))
    elif t <= 0.70:
        marker_fill = blend("#f59e0b", "#84cc16", (t - 0.45) / (0.70 - 0.45))
    else:
        marker_fill = blend("#84cc16", "#22c55e", (t - 0.70) / (1.00 - 0.70))

    description = (
        "Заглушка описания метрики. Здесь будет пояснение, что входит в техническое качество: "
        "чёткость изображения, качество звука, читаемость и общая аккуратность визуальной подачи."
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
                        stroke-dasharray="{value} 100"
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


def render_metric_bar(label: str, value: int, color: str) -> str:
    value = max(0, min(100, value))
    return f"""
    <div style="margin: 8px 0 10px 0; font-family:{FONT_STACK};">
        <div style="display:flex;justify-content:space-between;font-size:0.9rem;margin-bottom:4px;color:{TEXT_MAIN};">
            <span>{html.escape(label)}</span>
            <span><b>{value}</b></span>
        </div>
        <div style="width:100%;height:8px;background:{TRACK_BG};border-radius:999px;overflow:hidden;">
            <div style="width:{value}%;height:100%;background:{color};border-radius:999px;"></div>
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
        "metrics": metrics,
        "warnings": build_warning_flags(metrics),
    }
    return result


def render_running_panel():
    steps = [
        ("Получение и проверка ссылки", 15),
        ("Подготовка метаданных видео", 35),
        ("Извлечение признаков", 60),
        ("Формирование промежуточных оценок", 80),
        ("Подготовка итоговой выдачи", 100),
    ]

    current_step = st.session_state.progress_step
    step_text, step_value = steps[current_step]

    st.markdown("## Выполняется анализ")
    st.write("Система обрабатывает видео и подготавливает результат.")

    st.progress(step_value, text=step_text)
    st.markdown(f"### {step_text}")

    st.markdown("### Этапы анализа")
    for i, (name, _) in enumerate(steps):
        if i < current_step:
            st.success(f"✓ {name}")
        elif i == current_step:
            st.info(f"⏳ {name}")
        else:
            st.write(f"• {name}")

    time.sleep(1)

    if current_step < len(steps) - 1:
        st.session_state.progress_step += 1
        st.rerun()
    else:
        result = build_mock_result()

        st.session_state.current_result = result
        st.session_state.results_history.insert(0, result)
        st.session_state.results_history = st.session_state.results_history[:10]

        st.session_state.app_state = "done"
        st.session_state.has_result = True
        st.session_state.progress_step = 0
        st.rerun()


def render_result_panel(result: dict):
    st.markdown("## Результат анализа")

    video_url = result.get("video_url", "")
    if video_url:
        st.video(video_url)
    else:
        st.info("Превью видео недоступно.")

    st.markdown("### Краткое описание")
    st.write(result.get("summary", "Описание пока недоступно."))

    metrics = result.get("metrics", {})
    delivery_profile = metrics.get("delivery_profile", 18)
    tech_quality = metrics.get("tech_quality", 82)
    request_match = metrics.get("request_match", 74)
    warnings = result.get("warnings", build_warning_flags(metrics))

    st.markdown("### Ключевые метрики")

    outer_left, outer_center, outer_right = st.columns([1.08, 0.98, 1.08], gap="medium")

    with outer_left:
        left_content, left_spacer = st.columns([1, 0.08], gap="small")
        with left_content:
            render_bipolar_watch_widget(
                value=delivery_profile,
                title="Профиль подачи",
                left_label="Инструмент",
                right_label="Академия",
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

    render_match_bar(
        value=request_match,
        title="Соответствие запросу",
        subtitle="интегральная метрика соответствия видео выбранному сценарию",
    )


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

            bars_html = (
                render_metric_bar(
                    "Профиль подачи",
                    int((metrics.get("delivery_profile", 0) + 100) / 2),
                    ACCENT_VIOLET,
                )
                + render_metric_bar(
                    "Тех. качество",
                    metrics.get("tech_quality", 0),
                    ACCENT_GREEN,
                )
                + render_metric_bar(
                    "Соответствие",
                    metrics.get("request_match", 0),
                    ACCENT_CYAN,
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
        )

        st.divider()

        st.markdown("### Уровень погружённости")
        if "immersion_level" in st.session_state.form_errors:
            st.warning(f"❗ {st.session_state.form_errors['immersion_level']}")

        immersion_level = st.radio(
            "Уровень погружённости",
            options=IMMERSION_LEVELS,
            label_visibility="collapsed",
        )

        st.divider()

        st.markdown("### Цель просмотра")
        if "view_goal" in st.session_state.form_errors:
            st.warning(f"❗ {st.session_state.form_errors['view_goal']}")

        view_goal = st.radio(
            "Цель просмотра",
            options=VIEW_GOALS,
            label_visibility="collapsed",
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

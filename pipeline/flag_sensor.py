"""
Флаг «Сенсорик» — визуальный дискомфорт.

Компоненты:
  1. Qwen3-VL-Plus via DashScope API (Singapore) — 5 кадров, опрятность + слайды
  2. PySceneDetect — плотность склеек (cuts/min)
  3. OpenCV + MediaPipe — jerk score на каждой склейке
"""

from __future__ import annotations

import base64
import json
import logging
import os
import random
from dataclasses import dataclass, field, asdict
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ─── API ──────────────────────────────────────────────────────────────────────

QWEN_BASE_URL  = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
QWEN_MODEL     = "qwen3-vl-plus"
QWEN_ENV_KEY   = "DASHSCOPE_API_KEY"

# ─── Thresholds ───────────────────────────────────────────────────────────────

CUTS_PER_MINUTE_THRESHOLD = 4.0
JERK_SCORE_P90_THRESHOLD  = 0.35
VLM_FRAMES_COUNT          = 5
VLM_ISSUES_RATIO_THRESHOLD = 0.4   # ≥40% кадров с severity>=2 → флаг

CATEGORY_WEIGHT = {
    "overloaded_slide": 1.5,
    "visual_outburst":  1.3,
    "untidy":           1.0,
    "none":             0.0,
}

# ─── VLM prompt ───────────────────────────────────────────────────────────────

VLM_PROMPT = """\
Ты идентифицируешь визуальные проблемы в кадре лекции.
Только идентификация — НЕ оценивай степень тяжести.

КАТЕГОРИИ И ФАКТОРЫ:

UNTIDY (визуальная неаккуратность):
  U1 — предметы на фоне стоят вразнобой / не выровнены
       (книги, фигурки, куклы разной высоты как попало)
  U2 — одежда в непорядке (расстёгнута, мятая, перекошена, торчит воротник)
  U3 — РЕАЛЬНЫЙ blown-out пересвет, не просто светлая комната.
     Засчитывай ТОЛЬКО если:
     - на лице или объекте есть участок чисто-белого без деталей
     - явное гало/ореол вокруг головы или плеч от контрового света
     - очевидное пятно блика, мешающее различить лицо
     - заметные пятна, разводы, грязь в кадре
     НЕ U3: просто яркое освещение, белая стена, естественные блики на коже,
     светлая одежда, окно или экран проектора на фоне (если на лице нет засвета)
  U4 — что-то явно не на месте (провод через кадр, мусор, странно висящая/порванная штора)

OVERLOADED_SLIDE (перегрузка слайда):
  F1 — вычурный фон слайда: звёздное небо, галактика, космос, насыщенный градиент
     с переливами, текстура (мрамор, металл, блёстки), сложная фотография.
     Засчитывается ДАЖЕ если фон частично перекрыт текстом — важно, виден ли он.
     НЕ F1: однотонный цвет, один плавный градиент без объектов, белый/чёрный фон.
  F2 — несколько картинок/коллажей на слайде одновременно (2+)
  F3 — плотный многострочный текст
  F4 — контрастные цветные плашки/рамки, мешанина цветов
  F5 — мешанина шрифтов

VISUAL_OUTBURST (визуальный выхлест):
  V1 — неуместно яркий/контрастный объект перетягивает внимание с лектора
  V2 — экстравагантный образ лектора. Засчитывай ТОЛЬКО если ОДНОВРЕМЕННО
     присутствуют 2 и более маркеров:
     - крашеные волосы НЕЕСТЕСТВЕННЫХ цветов (синий, зелёный, розовый, фиолетовый).
       Натуральные цвета (рыжий, седой, чёрный, блонд) — НЕ маркер.
     - шляпа, бандана, корона, маска, костюмный головной убор
     - очевидно костюмная одежда (косплей, маскарад, исторический костюм)
     - неоновые / флуоресцентные / блестящие элементы
     - яркий грим, клоунский/театральный макияж
     - 3+ заметных аксессуара одновременно (цепи, крупный пирсинг, банты, перья)
     ПРЯМО НЕ ЯВЛЯЕТСЯ V2: обычная одежда любого цвета, надписи/принты на одежде,
     очки, борода, петличный микрофон, один яркий элемент образа.

ОСМЫСЛЕННЫЙ ДИЗАЙН ≠ БЕСПОРЯДОК. НЕ записывай в факторы:
- студийный фон с предметами, расставленными РОВНО и со смыслом
- однотонную яркую стену саму по себе
- предметы стоящие аккуратно и выровненно
- один декоративный объект (картина, цветок)

background_sharp: true — фон чёткий, объекты различимы; false — боке/расфокус
dominant_factor: true — фактор настолько груб, что доминирует над всем кадром

Выведи строго JSON, без markdown, без пояснений:
{
  "category": "none" | "untidy" | "overloaded_slide" | "visual_outburst",
  "factors": ["U1", "F2", ...],
  "background_sharp": true,
  "dominant_factor": false,
  "issues": ["короткое описание каждого фактора"],
  "noted_but_ok": ["что заметил, но это осмысленно/в норме"],
  "confidence": 0.85
}"""


# ─── Data structures ──────────────────────────────────────────────────────────

@dataclass
class SensorVLMResult:
    frame_results: list[dict] = field(default_factory=list)
    severity_scores: list[int] = field(default_factory=list)
    flag: bool = False
    issues_ratio: float = 0.0
    error: str | None = None


@dataclass
class SensorCutResult:
    cuts_per_minute: float
    total_cuts: int
    duration_sec: float
    flag: bool
    jerk_scores: list[float] = field(default_factory=list)
    jerk_p90: float = 0.0
    jerk_flag: bool = False


@dataclass
class SensorFlagResult:
    flag: bool
    confidence: float
    vlm: dict
    cuts: dict
    triggered_by: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


# ─── Severity computation ─────────────────────────────────────────────────────

def compute_severity(result: dict) -> int:
    factors  = list(result.get("factors", []))
    category = result.get("category", "none")
    bg_sharp = result.get("background_sharp", False)
    dominant = result.get("dominant_factor", False)

    # Одиночные визуальные факторы без dominant — часто ложные
    if factors in [["V2"], ["V1"]] and not dominant:
        return 0

    n = len(factors)
    weight = CATEGORY_WEIGHT.get(category, 1.0)
    severity = min(round(n * weight), 3)

    # Untidy виден только если фон резкий
    if category == "untidy" and not bg_sharp:
        severity = max(severity - 1, 0)

    if dominant:
        severity = max(severity, 2)

    return severity


# ─── VLM analysis ─────────────────────────────────────────────────────────────

def _encode_frame(path: str) -> str:
    """Кодируем кадр в base64 для API."""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _analyze_frame(client, frame_path: str) -> dict:
    """Отправляем один кадр в Qwen VL API."""
    b64 = _encode_frame(frame_path)
    ext = Path(frame_path).suffix.lstrip(".").lower()
    mime = f"image/{ext if ext in ('jpg', 'jpeg', 'png', 'webp') else 'jpeg'}"

    resp = client.chat.completions.create(
        model=QWEN_MODEL,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime};base64,{b64}"},
                },
                {"type": "text", "text": VLM_PROMPT},
            ],
        }],
        temperature=0.0,
        max_tokens=512,
    )

    raw = resp.choices[0].message.content.strip()
    # Убираем markdown-обёртку если есть
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return json.loads(raw.strip())


def _run_vlm(frame_paths: list[str]) -> SensorVLMResult:
    """Прогоняем 5 случайных кадров через Qwen VL."""
    api_key = os.environ.get(QWEN_ENV_KEY, "")
    if not api_key:
        logger.warning(f"{QWEN_ENV_KEY} не задан — VLM пропущен")
        return SensorVLMResult(error="no_api_key")

    if not frame_paths:
        return SensorVLMResult(error="no_frames")

    from openai import OpenAI
    client = OpenAI(api_key=api_key, base_url=QWEN_BASE_URL)

    # Берём 5 случайных кадров
    sampled = random.sample(frame_paths, min(VLM_FRAMES_COUNT, len(frame_paths)))

    frame_results = []
    severity_scores = []

    for path in sampled:
        try:
            result = _analyze_frame(client, path)
            severity = compute_severity(result)
            frame_results.append({**result, "frame": path, "severity": severity})
            severity_scores.append(severity)
            logger.debug(f"VLM: {Path(path).name} → category={result.get('category')} severity={severity}")
        except json.JSONDecodeError as e:
            logger.warning(f"VLM JSON parse error для {path}: {e}")
            severity_scores.append(0)
        except Exception as e:
            logger.warning(f"VLM ошибка для {path}: {e}")
            severity_scores.append(0)

    if not severity_scores:
        return SensorVLMResult(error="all_frames_failed")

    issues_ratio = sum(1 for s in severity_scores if s >= 2) / len(severity_scores)
    flag = (np.median(severity_scores) >= 2) or (np.percentile(severity_scores, 90) >= 2)

    return SensorVLMResult(
        frame_results=frame_results,
        severity_scores=severity_scores,
        flag=flag,
        issues_ratio=round(issues_ratio, 3),
    )


# ─── Cut analysis ─────────────────────────────────────────────────────────────

def _compute_jerk_score(
    frame_before: np.ndarray,
    frame_after: np.ndarray,
    pose=None,
) -> float:
    """
    Насколько резкая склейка:
      1. Histogram distance — общий визуальный сдвиг
      2. Brightness delta   — перепад яркости
      3. Pose jump          — смещение центроида плеч (если MediaPipe доступен)
    """
    scores = []

    # 1. Гистограмма
    hist_b = cv2.calcHist([frame_before], [0, 1, 2], None, [32, 32, 32], [0,256,0,256,0,256])
    hist_a = cv2.calcHist([frame_after],  [0, 1, 2], None, [32, 32, 32], [0,256,0,256,0,256])
    cv2.normalize(hist_b, hist_b)
    cv2.normalize(hist_a, hist_a)
    hist_dist = cv2.compareHist(hist_b, hist_a, cv2.HISTCMP_CHISQR)
    scores.append(min(hist_dist / 50.0, 1.0))

    # 2. Перепад яркости
    br_b = np.mean(cv2.cvtColor(frame_before, cv2.COLOR_BGR2GRAY))
    br_a = np.mean(cv2.cvtColor(frame_after,  cv2.COLOR_BGR2GRAY))
    scores.append(min(abs(br_a - br_b) / 100.0, 1.0))

    # 3. Pose jump — смещение лектора в кадре
    if pose is not None:
        try:
            import mediapipe as mp
            lm_idx = mp.solutions.pose.PoseLandmark

            res_b = pose.process(cv2.cvtColor(frame_before, cv2.COLOR_BGR2RGB))
            res_a = pose.process(cv2.cvtColor(frame_after,  cv2.COLOR_BGR2RGB))

            if res_b.pose_landmarks and res_a.pose_landmarks:
                cx_b = (res_b.pose_landmarks.landmark[lm_idx.LEFT_SHOULDER].x +
                        res_b.pose_landmarks.landmark[lm_idx.RIGHT_SHOULDER].x) / 2
                cx_a = (res_a.pose_landmarks.landmark[lm_idx.LEFT_SHOULDER].x +
                        res_a.pose_landmarks.landmark[lm_idx.RIGHT_SHOULDER].x) / 2
                # нормируем: прыжок на 30% ширины кадра = score 1.0
                scores.append(min(abs(cx_a - cx_b) / 0.3, 1.0))
        except Exception:
            pass  # если Pose упал — просто не добавляем компонент

    return float(np.mean(scores))


def _run_cuts(video_path: str) -> SensorCutResult:
    """PySceneDetect: плотность склеек + jerk score на каждой."""
    try:
        from scenedetect import detect, ContentDetector
    except ImportError:
        logger.warning("scenedetect не установлен")
        return SensorCutResult(
            cuts_per_minute=0, total_cuts=0, duration_sec=0,
            flag=False, jerk_flag=False,
        )

    scene_list = detect(video_path, ContentDetector(threshold=27.0))
    total_cuts  = max(len(scene_list) - 1, 0)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    duration_sec = total_frames / fps

    cuts_per_min = total_cuts / (duration_sec / 60.0) if duration_sec > 0 else 0.0

    # Jerk score на склейках
    jerk_scores = []

    # Инициализируем MediaPipe Pose один раз для всех склеек
    pose = None
    try:
        import mediapipe as mp
        pose = mp.solutions.pose.Pose(
            static_image_mode=True,
            min_detection_confidence=0.5,
        )
    except ImportError:
        logger.warning("mediapipe не установлен — pose_jump не считается")

    for i in range(len(scene_list) - 1):
        cut_frame_idx = int(scene_list[i][1].get_frames())

        cap.set(cv2.CAP_PROP_POS_FRAMES, max(cut_frame_idx - 2, 0))
        ret_b, frame_before = cap.read()
        cap.set(cv2.CAP_PROP_POS_FRAMES, cut_frame_idx + 2)
        ret_a, frame_after  = cap.read()

        if ret_b and ret_a:
            jerk = _compute_jerk_score(frame_before, frame_after, pose=pose)
            jerk_scores.append(jerk)

    if pose is not None:
        pose.close()

    cap.release()

    jerk_p90  = float(np.percentile(jerk_scores, 90)) if jerk_scores else 0.0
    jerk_flag = jerk_p90 > JERK_SCORE_P90_THRESHOLD
    cut_flag  = cuts_per_min > CUTS_PER_MINUTE_THRESHOLD

    return SensorCutResult(
        cuts_per_minute=round(cuts_per_min, 2),
        total_cuts=total_cuts,
        duration_sec=round(duration_sec, 1),
        flag=cut_flag,
        jerk_scores=[round(j, 4) for j in jerk_scores],
        jerk_p90=round(jerk_p90, 4),
        jerk_flag=jerk_flag,
    )


# ─── Main entry point ─────────────────────────────────────────────────────────

def run(
    video_path: str,
    frame_paths: list[str] | None = None,
) -> SensorFlagResult:
    """
    Запускает флаг Сенсорик.

    Args:
        video_path: путь к видеофайлу (для анализа склеек)
        frame_paths: уже извлечённые кадры для VLM (опционально;
                     если не переданы — извлекаем сами из video_path)

    Returns:
        SensorFlagResult
    """

    # Извлекаем кадры если не переданы
    if not frame_paths and video_path:
        logger.info("── Флаг Сенсорик: извлечение кадров ──")
        frame_paths = _extract_sample_frames(video_path, n=VLM_FRAMES_COUNT)

    logger.info("── Флаг Сенсорик: VLM анализ кадров ──")
    vlm = _run_vlm(frame_paths or [])

    logger.info("── Флаг Сенсорик: анализ склеек ──")
    cuts = _run_cuts(video_path)

    triggered = []
    if vlm.flag:    triggered.append(f"vlm_issues_ratio={vlm.issues_ratio:.2f}")
    if cuts.flag:   triggered.append(f"cuts_per_min={cuts.cuts_per_minute:.1f}")
    if cuts.jerk_flag: triggered.append(f"jerk_p90={cuts.jerk_p90:.3f}")

    flag = vlm.flag or cuts.flag or cuts.jerk_flag

    # Уверенность
    confidence_parts = []
    if vlm.severity_scores:
        confidence_parts.append(min(vlm.issues_ratio * 2, 1.0))
    if cuts.flag or cuts.jerk_flag:
        confidence_parts.append(0.9)
    confidence = round(float(np.mean(confidence_parts)) if confidence_parts else 0.0, 3)

    return SensorFlagResult(
        flag=flag,
        confidence=confidence,
        vlm={
            "flag":          vlm.flag,
            "issues_ratio":  vlm.issues_ratio,
            "severity_scores": vlm.severity_scores,
            "frame_results": vlm.frame_results,
            "error":         vlm.error,
        },
        cuts={
            "cuts_per_minute":      cuts.cuts_per_minute,
            "total_cuts":           cuts.total_cuts,
            "duration_sec":         cuts.duration_sec,
            "cut_density_flag":     cuts.flag,
            "jerk_p90":             cuts.jerk_p90,
            "jerk_flag":            cuts.jerk_flag,
            "threshold_cuts_pm":    CUTS_PER_MINUTE_THRESHOLD,
            "threshold_jerk_p90":   JERK_SCORE_P90_THRESHOLD,
        },
        triggered_by=triggered,
    )


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _extract_sample_frames(video_path: str, n: int = 5) -> list[str]:
    """Извлекаем n равномерно распределённых кадров из видео."""
    import tempfile

    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps   = cap.get(cv2.CAP_PROP_FPS) or 25.0

    # Пропускаем первые и последние 5% — там часто заставки
    start = int(total * 0.05)
    end   = int(total * 0.95)
    indices = np.linspace(start, end, n, dtype=int).tolist()

    out_dir = tempfile.mkdtemp(prefix="sensor_frames_")
    paths = []

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        path = os.path.join(out_dir, f"frame_{idx:06d}.jpg")
        cv2.imwrite(path, frame)
        paths.append(path)

    cap.release()
    return paths
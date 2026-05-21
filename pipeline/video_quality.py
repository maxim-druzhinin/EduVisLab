"""
Модуль технического качества видео.

Метрики:
  - Технические параметры (ffprobe) — разрешение, fps, битрейт
  - DOVER (technical + aesthetic) — перцептивное качество, аналог DNSMOS
  - BRISQUE — артефакты сжатия и зерно
  - Blur (Лапласиан) — резкость / размытость
  - Экспозиция — яркость, пересвет, недосвет
  - Контраст — диапазон яркостей
  - Flicker — мерцание от ламп
  - Стабильность (vidstabdetect) — дрожание камеры
  - Временная консистентность — std метрик по всему видео
  - Видимость спикера (MediaPipe) — лицо в кадре
  - Читаемость доски / экрана — засветка + локальный контраст

Входные данные: путь к скачанному видеофайлу.
"""

import os
import logging
import subprocess
import json
import tempfile
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

import base64
from openai import OpenAI

from config import (
    RESOLUTION_HIGH, RESOLUTION_NORMAL, RESOLUTION_LOW,
    DOVER_GOOD, DOVER_BAD,
    BRISQUE_GOOD, BRISQUE_BAD,
    BLUR_SHARP, BLUR_SOFT,
    BRIGHTNESS_MIN, BRIGHTNESS_MAX,
    OVEREXPOSED_RATIO_OK, UNDEREXPOSED_RATIO_OK,
    CONTRAST_GOOD, CONTRAST_OK,
    FLICKER_MILD, FLICKER_SEVERE,
    STABILITY_OK, STABILITY_SHAKY, STABILITY_VERY_SHAKY,
    CONSISTENCY_MODERATE, CONSISTENCY_INCONSISTENT,
    FACE_PRESENCE_HIGH, FACE_PRESENCE_LOW,
    FACE_SIZE_OK, FACE_SIZE_SMALL,
    BOARD_GLARE_PARTIAL, BOARD_GLARE_SEVERE,
    BOARD_CONTRAST_READABLE, BOARD_CONTRAST_MARGINAL,
    FRAMES_DIR, DEVICE
)

logger = logging.getLogger(__name__)


# ─── Data classes ─────────────────────────────────────────────────────────────

@dataclass
class VideoQualityResult:
    # Технические параметры
    width: int
    height: int
    fps: float
    bitrate_kbps: int
    codec: str
    resolution_level: str           # "high" / "normal" / "low" / "bad"

    # Перцептивное качество (DOVER)
    dover_score: Optional[float]   # fused 0..1
    dover_quality: str                 # "good" / "ok" / "bad" / "unavailable"

    # Артефакты / зерно (BRISQUE)
    brisque_median: float
    brisque_p90: float
    brisque_level: str              # "good" / "ok" / "bad"

    # Резкость (Лапласиан)
    blur_median: float
    blur_p10: float
    blur_level: str                 # "sharp" / "soft" / "blurry"

    # Освещение
    mean_brightness: float
    bright_background_ratio: float
    overexposed_ratio: float
    underexposed_ratio: float
    exposure_level: str             # "normal" / "dark" / "overexposed" / "underexposed"

    # Контраст
    contrast_median: float
    contrast_level: str             # "good" / "ok" / "low"

    # Мерцание
    flicker_mean: float
    flicker_level: str              # "none" / "mild" / "severe"

    # Стабильность (vidstabdetect)
    motion_mean: float
    motion_std: float
    stability_level: str            # "stable" / "ok" / "shaky" / "very_shaky"

    # Временная консистентность
    blur_std: float
    brisque_std: float
    dover_technical_std: Optional[float]
    brightness_std: float
    consistency_level: str          # "consistent" / "moderate" / "inconsistent"

    # Видимость спикера
    face_presence_ratio: float
    face_size_median: float

    # Читаемость доски / экрана
    board_detected: bool
    board_glare_ratio: Optional[float]
    board_contrast: Optional[float]
    board_glare_level: Optional[str]        # "ok" / "partial" / "severe"
    # VLM оценка
    board_readability_level: Optional[int]    # 1-5, ← было str
    board_readability_score: Optional[float]  # 0..1, ← новое
    board_surface_type: Optional[str]         # ← новое
    board_main_issues: Optional[list] 


QWEN_BOARD_PROMPT = """
Тебе показан кадр из образовательного видео. YOLOE уже детектировал 
на этом кадре визуальный носитель контента — оцени насколько хорошо 
видно то что на нём написано или показано.

Носитель может быть любым: классная доска (зелёная или чёрная), 
маркерная доска, проекционный экран со слайдами, флипчарт, 
монитор или планшет в кадре. Подход к оценке одинаковый для всех.

ЧТО ОЦЕНИВАЕМ — только физические условия восприятия:
- равномерность и качество освещения области носителя
- контраст между контентом и фоном
- чёткость и размер контента относительно камеры
- плотность контента

ЧТО НЕ ОЦЕНИВАЕМ:
- язык, алфавит, формулы — нас не интересует что написано/показано, 
  только видно ли это физически
- тип контента — рукопись, печатный текст и слайды оцениваются 
  одинаково, не занижай оценку за нечеловеческий шрифт
- временное перекрытие лектором — если лектор стоит перед доской, 
  оценивай только видимую часть
- качество почерка — кривой но контрастный почерк это уровень 4-5

Шкала:
1 — НЕЧИТАЕМО: засветка, глубокая тень или размытость уничтожают весь контент
2 — СЛОЖНО ЧИТАТЬ: значительная часть контента теряется из-за освещения или контраста
3 — ЧАСТИЧНО ЧИТАЕМО: общая картина понятна, детали теряются
4 — ХОРОШО ЧИТАЕМО: контент воспринимается без затруднений, незначительные проблемы
5 — ОТЛИЧНО ЧИТАЕМО: равномерное освещение, высокий контраст, всё чётко различимо

Ответь строго в JSON без каких-либо пояснений вне JSON:
{
  "readability_level": 1-5,
  "readability_score": 0.0-1.0,
  "surface_type": "blackboard/whiteboard/screen/flipchart/tablet/monitor",
  "main_issue": "главная проблема одним словом или null",
  "evidence": "одно конкретное наблюдение из кадра"
}
"""


# ─── Step 0: Download video ───────────────────────────────────────────────────

def download_video(url: str, output_dir: str = "/tmp/pipeline_video") -> str:
    """
    Скачивает видеофайл с YouTube через yt-dlp.
    В отличие от transcription.py — скачиваем видео, не только аудио.
    Возвращает путь к скачанному файлу.
    """
    os.makedirs(output_dir, exist_ok=True)
    output_template = os.path.join(output_dir, "%(id)s.%(ext)s")

    cmd = [
        "yt-dlp",
        "--no-playlist",
        "--format", "bestvideo[height<=1080][vcodec^=avc1][ext=mp4]/bestvideo[height<=1080][ext=mp4]/bestvideo[height<=1080]",
        "--output", output_template,
        "--print", "after_move:filepath",
        url,
    ]

    logger.info(f"Скачиваем видео: {url}")
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)

    video_path = result.stdout.strip().splitlines()[-1]
    logger.info(f"Видео скачано: {video_path}")
    return video_path


# ─── Step 1: Adaptive frame extraction ───────────────────────────────────────

def get_hwaccel_args() -> list:
    return ["-hwaccel", "cuda"] if DEVICE == "cuda" else []

def extract_quality_frames(video_path: str, fps: float = 0.1) -> list[np.ndarray]:
    """
    Извлекает кадры с низкой частотой (0.1 fps = 1 кадр / 10 сек).
    Для DOVER, BRISQUE, blur, экспозиции, контраста, видимости спикера.
    """
    frames_dir = os.path.join(FRAMES_DIR, "quality")
    os.makedirs(frames_dir, exist_ok=True)

    pattern = os.path.join(frames_dir, "frame_%04d.jpg")
    cmd = [
        "ffmpeg", "-y",
        *get_hwaccel_args(),  # пусто если CUDA нет
        "-i", video_path,
        "-vf", f"fps={fps}",
        "-q:v", "2",
        pattern,
    ]
    subprocess.run(cmd, capture_output=True, check=True)

    frame_paths = sorted(Path(frames_dir).glob("frame_*.jpg"))
    frames = [cv2.imread(str(p)) for p in frame_paths]
    frames = [f for f in frames if f is not None]

    logger.info(f"Quality frames: {len(frames)} кадров при {fps} fps")
    return frames


def extract_motion_frames(
    video_path: str,
    n_windows: int = 5,
    window_duration: int = 10,
    target_fps: int = 8,
) -> list[np.ndarray]:
    """
    Извлекает кадры с высокой частотой (8 fps) из случайных окон.
    Для мерцания (flicker). Видео длиной < window_duration — берём целиком.

    vidstabdetect работает напрямую с видеофайлом — эти кадры только для flicker.
    """
    # Получаем длительность видео
    probe = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json",
         "-show_format", video_path],
        capture_output=True, text=True,
    )
    duration = float(json.loads(probe.stdout)["format"]["duration"])

    frames_dir = os.path.join(FRAMES_DIR, "motion")
    os.makedirs(frames_dir, exist_ok=True)

    all_frames = []

    if duration <= window_duration:
        # Короткое видео — берём целиком
        windows = [(0, duration)]
    else:
        # Случайные окна, но не в самом начале (титры) и конце (аутро)
        margin = min(30, duration * 0.1)
        safe_start = margin
        safe_end   = duration - margin - window_duration
        if safe_end <= safe_start:
            safe_end = duration - window_duration

        starts = random.sample(
            range(int(safe_start), max(int(safe_end), int(safe_start) + 1)),
            min(n_windows, max(1, int(safe_end - safe_start)))
        )
        windows = [(s, s + window_duration) for s in starts]

    for i, (start, end) in enumerate(windows):
        window_dir = os.path.join(frames_dir, f"window_{i:02d}")
        os.makedirs(window_dir, exist_ok=True)
        pattern = os.path.join(window_dir, "frame_%04d.jpg")

        cmd = [
            "ffmpeg", "-y",
            "-ss", str(start),
            "-to", str(end),
            "-i", video_path,
            "-vf", f"fps={target_fps}",
            "-q:v", "2",
            pattern,
        ]
        subprocess.run(cmd, capture_output=True, check=True)

        window_frames = sorted(Path(window_dir).glob("frame_*.jpg"))
        for fp in window_frames:
            frame = cv2.imread(str(fp))
            if frame is not None:
                all_frames.append(frame)

    logger.info(f"Motion frames: {len(all_frames)} кадров из {len(windows)} окон")
    return all_frames


# ─── Step 2: ffprobe metadata ─────────────────────────────────────────────────

def get_video_metadata(video_path: str) -> dict:
    cmd = [
        "ffprobe", "-v", "quiet",
        "-print_format", "json",
        "-show_streams",
        "-show_format",  # ← добавили
        video_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    data = json.loads(result.stdout)

    video_stream = next(
        (s for s in data["streams"] if s["codec_type"] == "video"),
        None,
    )
    if not video_stream:
        raise RuntimeError("Видеопоток не найден в файле")

    fps_raw = video_stream.get("avg_frame_rate", "0/1")
    try:
        fps = eval(fps_raw)  # noqa: S307
    except Exception:
        fps = 0.0

    # Stream битрейт часто занижен — берём format как fallback
    stream_bitrate = int(video_stream.get("bit_rate") or 0)
    format_bitrate = int(data["format"].get("bit_rate") or 0)
    bitrate = format_bitrate if stream_bitrate < 1000 else stream_bitrate

    meta = {
        "width":        int(video_stream["width"]),
        "height":       int(video_stream["height"]),
        "fps":          round(float(fps), 2),
        "bitrate_kbps": bitrate // 1000,
        "codec":        video_stream.get("codec_name", "unknown"),
    }

    logger.info(
        f"Метаданные: {meta['width']}x{meta['height']} "
        f"{meta['fps']}fps {meta['codec']} {meta['bitrate_kbps']}kbps"
    )
    return meta


def interpret_resolution(height: int) -> str:
    if height >= RESOLUTION_HIGH:
        return "high"
    elif height >= RESOLUTION_NORMAL:
        return "normal"
    elif height >= RESOLUTION_LOW:
        return "low"
    return "bad"


# ─── Step 3: DOVER ────────────────────────────────────────────────────────────

def compute_dover(video_path: str) -> Optional[float]:
    import subprocess
    try:
        result = subprocess.run(
            ["python", "evaluate_one_video.py", "-v", video_path, "-f"],
            cwd="/tmp/DOVER",
            capture_output=True,
            text=True,
        )
        line = result.stdout.strip().splitlines()[-1]
        fused = float(line.split(":")[-1].strip())
        logger.info(f"DOVER: {fused}")
        return round(fused, 4)
    except Exception as e:
        logger.warning(f"DOVER ошибка: {e}")
        return None


def interpret_dover(score: Optional[float]) -> str:
    if score is None:
        return "unavailable"
    if score >= DOVER_GOOD:
        return "good"
    elif score >= DOVER_BAD:
        return "ok"
    return "bad"


# ─── Step 4: BRISQUE ──────────────────────────────────────────────────────────

def compute_brisque_scores(frames: list[np.ndarray]) -> list[float]:
    import torch
    from piq import brisque

    logger.info("Считаем BRISQUE...")

    BATCH_SIZE = 32
    device = torch.device(DEVICE)

    # Конвертируем все кадры: resize до 512px + на device
    tensors = []
    for frame in frames:
        rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (512, 512))
        tensor  = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0
        tensors.append(tensor)

    scores = []
    for i in range(0, len(tensors), BATCH_SIZE):
        batch = torch.stack(tensors[i:i + BATCH_SIZE]).to(device)
        try:
            batch_scores = brisque(batch, data_range=1.0, reduction='none')
            scores.extend(batch_scores.cpu().tolist())
        except Exception as e:
            logger.warning(f"BRISQUE batch {i}: {e}")

    logger.info(f"BRISQUE: {len(scores)} кадров обработано")
    return scores


def interpret_brisque(median: float) -> str:
    if median < BRISQUE_GOOD:
        return "good"
    elif median < BRISQUE_BAD:
        return "ok"
    return "bad"


# ─── Step 5: Blur (Laplacian) ─────────────────────────────────────────────────

def blur_score(frame: np.ndarray) -> float:
    """Дисперсия Лапласиана. Выше = резче."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def compute_blur_scores(frames: list[np.ndarray]) -> list[float]:
    logger.info("Считаем blur (Лапласиан)...")
    scores = [blur_score(f) for f in frames]
    logger.info(f"Blur: median={np.median(scores):.1f}, p10={np.percentile(scores, 10):.1f}")
    return scores


def interpret_blur(p10: float) -> str:
    if p10 >= BLUR_SHARP:
        return "sharp"
    elif p10 >= BLUR_SOFT:
        return "soft"
    return "blurry"


# ─── Step 6: Exposure ─────────────────────────────────────────────────────────

def exposure_metrics(frame: np.ndarray) -> dict:
    hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    v    = hsv[:, :, 2].astype(np.float32) / 255.0

    bright_mask  = v > 0.95
    bright_ratio = float(np.mean(bright_mask))
    under_ratio  = float(np.mean(v < 0.05))

    real_overexposed_ratio = 0.0
    bright_edge_density    = 0.0

    if bright_mask.sum() > 100:
        lap   = cv2.Laplacian(gray, cv2.CV_64F)
        edges = np.abs(lap) > 12.0
        bright_edge_density = float(np.mean(edges[bright_mask]))
        if bright_edge_density < 0.01:
            real_overexposed_ratio = bright_ratio

    return {
        "mean_brightness":          float(np.mean(v)),
        "bright_background_ratio":  bright_ratio,           # белый фон (доска/слайды)
        "overexposed_ratio":        real_overexposed_ratio, # реальный пересвет
        "bright_edge_density":      bright_edge_density,
        "underexposed_ratio":       under_ratio,
    }


def compute_exposure(frames: list[np.ndarray]) -> dict:
    logger.info("Считаем экспозицию...")
    all_metrics = [exposure_metrics(f) for f in frames]

    mean_brightness    = float(np.median([m["mean_brightness"]    for m in all_metrics]))
    overexposed_ratio  = float(np.median([m["overexposed_ratio"]  for m in all_metrics]))
    underexposed_ratio = float(np.median([m["underexposed_ratio"] for m in all_metrics]))

    logger.info(
        f"Экспозиция: brightness={mean_brightness:.3f}, "
        f"over={overexposed_ratio:.4f}, under={underexposed_ratio:.4f}"
    )
    return {
        "mean_brightness":    round(mean_brightness, 3),
        "overexposed_ratio":  round(overexposed_ratio, 4),
        "underexposed_ratio": round(underexposed_ratio, 4),
    }


def interpret_exposure(mean_brightness: float, overexposed: float, underexposed: float) -> str:
    if overexposed > OVEREXPOSED_RATIO_OK * 5:
        return "overexposed"
    if underexposed > UNDEREXPOSED_RATIO_OK * 4:
        return "underexposed"
    if mean_brightness < BRIGHTNESS_MIN:
        return "dark"
    if mean_brightness > BRIGHTNESS_MAX:
        return "overexposed"
    return "normal"


# ─── Step 7: Contrast ─────────────────────────────────────────────────────────

def contrast_score_frame(frame: np.ndarray) -> float:
    """Стандартное отклонение яркости — мера контраста. Выше = контрастнее."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(float) / 255.0
    return float(np.std(gray))


def compute_contrast(frames: list[np.ndarray]) -> float:
    logger.info("Считаем контраст...")
    scores = [contrast_score_frame(f) for f in frames]
    median = round(float(np.median(scores)), 4)
    logger.info(f"Контраст median: {median:.4f}")
    return median


def interpret_contrast(median: float) -> str:
    if median >= CONTRAST_GOOD:
        return "good"
    elif median >= CONTRAST_OK:
        return "ok"
    return "low"


# ─── Step 8: Flicker ──────────────────────────────────────────────────────────

def compute_flicker(motion_frames: list[np.ndarray]) -> dict:
    """
    Мерцание по кадрам высокой частоты (8 fps).
    Детектирует биение флуоресцентных ламп (50/100 Гц).
    """
    logger.info("Считаем flicker...")

    if len(motion_frames) < 2:
        logger.warning("Flicker: недостаточно кадров")
        return {"flicker_mean": 0.0, "flicker_std": 0.0}

    brightness = []
    for frame in motion_frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness.append(float(np.mean(gray)))

    diffs = np.abs(np.diff(brightness))
    flicker_mean = round(float(np.mean(diffs)), 4)
    flicker_std  = round(float(np.std(diffs)), 4)

    logger.info(f"Flicker: mean={flicker_mean:.4f}, std={flicker_std:.4f}")
    return {"flicker_mean": flicker_mean, "flicker_std": flicker_std}


def interpret_flicker(mean: float) -> str:
    if mean < FLICKER_MILD:
        return "none"
    elif mean < FLICKER_SEVERE:
        return "mild"
    return "severe"


# ─── Step 9: Stability (vidstabdetect) ───────────────────────────────────────

def compute_stability(video_path: str) -> dict:
    import re

    probe = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", video_path],
        capture_output=True, text=True,
    )
    duration = float(json.loads(probe.stdout)["format"]["duration"])

    margin = min(30, duration * 0.1)
    windows = [
        random.uniform(margin, duration - 10 - margin)
        for _ in range(3)
    ]

    all_content = ""

    for i, start in enumerate(windows):
        trf_path = os.path.join(tempfile.gettempdir(), f"stability_{i}.trf")
        try:
            subprocess.run([
                "ffmpeg", "-y",
                "-ss", str(start),
                "-t", "10",
                "-i", video_path,
                "-vf", f"scale=640:-1,vidstabdetect=result={trf_path}:shakiness=5:accuracy=9",
                "-f", "null", "-",
            ], capture_output=True, check=True)

            with open(trf_path) as f:
                all_content += f.read()

        except Exception as e:
            logger.warning(f"Stability окно {i}: {e}")

    if not all_content:
        return {"motion_mean": 0.0, "motion_std": 0.0}

    # Медиана по кадрам с фильтрацией по match score
    # Формат LM: dx dy x y size contrast match
    frame_medians = []
    frames = re.findall(r'Frame\s+\d+.*?(?=Frame|\Z)', all_content, re.DOTALL)

    for frame_content in frames:
        lms = re.findall(
            r'LM\s+(-?\d+)\s+(-?\d+)\s+\d+\s+\d+\s+\d+\s+[\d.]+\s+([\d.]+)',
            frame_content
        )
        # match < 0.3 — хорошее совпадение блоков, не склейка
        good = [(int(dx), int(dy)) for dx, dy, match in lms if float(match) < 0.3]
        if good:
            mags = [(dx**2 + dy**2)**0.5 for dx, dy in good]
            frame_medians.append(float(np.median(mags)))

    if not frame_medians:
        return {"motion_mean": 0.0, "motion_std": 0.0}

    motion_mean = round(float(np.mean(frame_medians)), 3)
    motion_std  = round(float(np.std(frame_medians)), 3)

    logger.info(f"Стабильность: motion_mean={motion_mean}, std={motion_std}")
    return {"motion_mean": motion_mean, "motion_std": motion_std}


def interpret_stability(motion_mean: float) -> str:
    if motion_mean < STABILITY_OK:
        return "stable"
    elif motion_mean < STABILITY_SHAKY:
        return "ok"
    elif motion_mean < STABILITY_VERY_SHAKY:
        return "shaky"
    return "very_shaky"


# ─── Step 10: Temporal consistency ───────────────────────────────────────────

def compute_temporal_consistency(
    blur_scores: list[float],
    brisque_scores: list[float],
    dover_scores: list[float],
    brightness_scores: list[float],
) -> dict:
    """
    std покадровых скоров — мера непостоянства качества по ходу видео.
    Высокий std → качество скачет (например изменились условия освещения).

    Интерпретируем через relative std = std / mean.
    """
    def rel_std(values: list[float]) -> float:
        if not values:
            return 0.0
        mean = float(np.mean(values))
        std  = float(np.std(values))
        return round(std / mean if mean > 0 else 0.0, 4)

    result = {
        "blur_std":      round(float(np.std(blur_scores)), 3) if blur_scores else 0.0,
        "brisque_std":   round(float(np.std(brisque_scores)), 3) if brisque_scores else 0.0,
        "brightness_std": round(float(np.std(brightness_scores)), 3) if brightness_scores else 0.0,
        "blur_rel_std":  rel_std(blur_scores),
    }

    if dover_scores:
        result["dover_technical_std"] = round(float(np.std(dover_scores)), 3)
    else:
        result["dover_technical_std"] = None

    logger.info(
        f"Консистентность: blur_std={result['blur_std']:.3f}, "
        f"brisque_std={result['brisque_std']:.3f}"
    )
    return result


def interpret_consistency(blur_rel_std: float) -> str:
    """Используем blur как основной индикатор — наиболее стабильная метрика."""
    if blur_rel_std < CONSISTENCY_MODERATE:
        return "consistent"
    elif blur_rel_std < CONSISTENCY_INCONSISTENT:
        return "moderate"
    return "inconsistent"


# ─── Step 11: Face visibility (MediaPipe) ────────────────────────────────────

def compute_face_visibility(frames: list[np.ndarray]) -> dict:
    import mediapipe as mp

    logger.info("Считаем видимость спикера (MediaPipe)...")

    BaseOptions     = mp.tasks.BaseOptions
    FaceDetector    = mp.tasks.vision.FaceDetector
    FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
    VisionRunningMode   = mp.tasks.vision.RunningMode

    options = FaceDetectorOptions(
        base_options=BaseOptions(model_asset_path="/tmp/face_detector.tflite"),
        running_mode=VisionRunningMode.IMAGE,
    )

    face_sizes    = []
    frames_w_face = 0

    with FaceDetector.create_from_options(options) as detector:
        for frame in frames:
            rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result   = detector.detect(mp_image)

            if result.detections:
                frames_w_face += 1
                bbox = result.detections[0].bounding_box
                h, w = frame.shape[:2]
                face_area = (bbox.width * bbox.height) / (w * h)
                face_sizes.append(face_area)

    face_presence_ratio = round(frames_w_face / len(frames), 3) if frames else 0.0
    face_size_median    = round(float(np.median(face_sizes)), 4) if face_sizes else 0.0

    logger.info(f"Спикер: presence={face_presence_ratio:.3f}, size_median={face_size_median:.4f}")
    return {
        "face_presence_ratio": face_presence_ratio,
        "face_size_median":    face_size_median,
    }


# ─── Step 12: Board readability ───────────────────────────────────────────────

def detect_board_masks(frames: list[np.ndarray]) -> list[Optional[np.ndarray]]:
    """
    Детектирует область доски/экрана через YOLOE-26x-seg.
    Возвращает список масок (None если доска не найдена на кадре).
    """
    from ultralytics import YOLOE

    logger.info("Детектируем доску через YOLOE...")
    model = YOLOE("yoloe-26x-seg.pt")
    model.set_classes(
        ["whiteboard", "blackboard", "chalkboard",
         "projection screen", "flipchart", "monitor", "tablet"]
    )

    masks = []
    for frame in frames:
        results = model.predict(frame, verbose=False)
        if results[0].masks is not None and len(results[0].masks) > 0:
            mask = results[0].masks.data[0].cpu().numpy().astype(bool)
            # Ресайзим маску до размера кадра если нужно
            if mask.shape != frame.shape[:2]:
                mask = cv2.resize(
                    mask.astype(np.uint8),
                    (frame.shape[1], frame.shape[0])
                ).astype(bool)
            masks.append(mask)
        else:
            masks.append(None)

    detected = sum(1 for m in masks if m is not None)
    logger.info(f"Доска найдена на {detected}/{len(frames)} кадрах")
    return masks


def compute_board_cv_metrics(
    frames: list[np.ndarray],
    masks: list[Optional[np.ndarray]],
) -> dict:
    """CV метрики по области доски: засветка и локальный контраст."""
    glare_scores = []
    contrast_scores = []

    for frame, mask in zip(frames, masks):
        if mask is None:
            continue
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        v   = hsv[:, :, 2].astype(float) / 255.0
        glare_scores.append(float(np.mean(v[mask] > 0.92)))

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(float) / 255.0
        contrast_scores.append(float(np.std(gray[mask])))

    if not glare_scores:
        return {
            "board_detected": False,
            "board_glare_ratio": None,
            "board_contrast": None,
            "board_glare_level": None,
        }

    glare_median    = round(float(np.median(glare_scores)), 4)
    contrast_median = round(float(np.median(contrast_scores)), 4)

    if glare_median >= BOARD_GLARE_SEVERE:
        glare_level = "severe"
    elif glare_median >= BOARD_GLARE_PARTIAL:
        glare_level = "partial"
    else:
        glare_level = "ok"

    return {
        "board_detected":    True,
        "board_glare_ratio": glare_median,
        "board_contrast":    contrast_median,
        "board_glare_level": glare_level,
    }


def compute_board_readability_vlm(
    frames: list[np.ndarray],
    masks: list[Optional[np.ndarray]],
) -> dict:
    """VLM оценка читаемости через Qwen3-VL-Plus. Берём 5 кадров где есть доска."""
    api_key = os.environ.get("DASHSCOPE_API_KEY")
    if not api_key:
        logger.warning("DASHSCOPE_API_KEY не задан — пропускаем VLM оценку доски")
        return {
            "board_readability_level": None,
            "board_readability_score": None,
            "board_surface_type":      None,
            "board_main_issues":       None,
        }

    client = OpenAI(
        api_key=api_key,
        base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
    )

    valid = [(f, m) for f, m in zip(frames, masks) if m is not None]
    if not valid:
        return {
            "board_readability_level": None,
            "board_readability_score": None,
            "board_surface_type":      None,
            "board_main_issues":       None,
        }

    indices  = np.linspace(0, len(valid) - 1, min(5, len(valid)), dtype=int)
    selected = [valid[i] for i in indices]

    results = []
    for frame, mask in selected:
        rows, cols = np.where(mask)
        if len(rows) == 0:
            continue
        crop = frame[rows.min():rows.max(), cols.min():cols.max()]

        _, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
        img_b64 = base64.b64encode(buf).decode("utf-8")

        try:
            response = client.chat.completions.create(
                model="qwen3-vl-plus",
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image_url",
                         "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
                        {"type": "text", "text": QWEN_BOARD_PROMPT},
                    ],
                }],
                max_tokens=200,
            )
            raw = response.choices[0].message.content.strip()
            raw = raw[raw.find("{"):raw.rfind("}") + 1]
            parsed = json.loads(raw)
            results.append(parsed)
            logger.info(f"Qwen board: level={parsed.get('readability_level')} "
                       f"surface={parsed.get('surface_type')} "
                       f"issue={parsed.get('main_issue')}")
        except Exception as e:
            logger.warning(f"Qwen board API error: {e}")

    if not results:
        return {
            "board_readability_level": None,
            "board_readability_score": None,
            "board_surface_type":      None,
            "board_main_issues":       None,
        }

    levels   = [r["readability_level"] for r in results if "readability_level" in r]
    scores   = [r["readability_score"]  for r in results if "readability_score"  in r]
    issues   = [r["main_issue"] for r in results if r.get("main_issue")]
    surfaces = [r["surface_type"] for r in results if r.get("surface_type")]

    return {
        "board_readability_level": int(np.median(levels)) if levels else None,
        "board_readability_score": round(float(np.mean(scores)), 3) if scores else None,
        "board_surface_type":      max(set(surfaces), key=surfaces.count) if surfaces else None,
        "board_main_issues":       list(set(issues)) if issues else None,
    }


# ─── Main entry point ─────────────────────────────────────────────────────────

def run(video_path: str) -> VideoQualityResult:
    """
    Полный анализ технического качества видео.
    Принимает путь к скачанному видеофайлу.
    """
    logger.info(f"Анализируем качество видео: {video_path}")

    # ── Метаданные (ffprobe) ───────────────────────────────────────────────
    meta = get_video_metadata(video_path)

    # ── Извлечение кадров ─────────────────────────────────────────────────
    quality_frames = extract_quality_frames(video_path, fps=0.1)
    motion_frames  = extract_motion_frames(video_path)

    if not quality_frames:
        raise RuntimeError("Не удалось извлечь кадры из видео")

    # ── DOVER ─────────────────────────────────────────────────────────────
    dover_score = compute_dover(video_path)

    # ── BRISQUE ───────────────────────────────────────────────────────────
    brisque_scores = compute_brisque_scores(quality_frames)
    brisque_median = round(float(np.median(brisque_scores)), 2) if brisque_scores else 0.0
    brisque_p90    = round(float(np.percentile(brisque_scores, 90)), 2) if brisque_scores else 0.0

    # ── Blur ──────────────────────────────────────────────────────────────
    blur_scores = compute_blur_scores(quality_frames)
    blur_median = round(float(np.median(blur_scores)), 2)
    blur_p10    = round(float(np.percentile(blur_scores, 10)), 2)

    # ── Экспозиция ────────────────────────────────────────────────────────
    exposure = compute_exposure(quality_frames)

    # ── Контраст ──────────────────────────────────────────────────────────
    contrast_median = compute_contrast(quality_frames)

    # ── Flicker ───────────────────────────────────────────────────────────
    flicker = compute_flicker(motion_frames)

    # ── Стабильность ──────────────────────────────────────────────────────
    stability = compute_stability(video_path)

    # ── Временная консистентность ─────────────────────────────────────────
    brightness_scores = [
        exposure_metrics(f)["mean_brightness"] for f in quality_frames
    ]
    consistency = compute_temporal_consistency(
        blur_scores, brisque_scores,
        [],  # dover_scores — больше не собираем покадрово
        brightness_scores,
    )

    # ── Видимость спикера ─────────────────────────────────────────────────
    face = compute_face_visibility(quality_frames)

    # ── Читаемость доски ──────────────────────────────────────────────────
    board_masks = detect_board_masks(quality_frames)
    board_cv    = compute_board_cv_metrics(quality_frames, board_masks)
    board_vlm   = compute_board_readability_vlm(quality_frames, board_masks)

    logger.info("── Качество видео готово ──")

    return VideoQualityResult(
        # Технические параметры
        width=meta["width"],
        height=meta["height"],
        fps=meta["fps"],
        bitrate_kbps=meta["bitrate_kbps"],
        codec=meta["codec"],
        resolution_level=interpret_resolution(meta["height"]),

        # DOVER
        dover_score=dover_score,
        dover_quality=interpret_dover(dover_score),

        # BRISQUE
        brisque_median=brisque_median,
        brisque_p90=brisque_p90,
        brisque_level=interpret_brisque(brisque_median),

        # Blur
        blur_median=blur_median,
        blur_p10=blur_p10,
        blur_level=interpret_blur(blur_p10),

        # Экспозиция
        mean_brightness=exposure["mean_brightness"],
        bright_background_ratio=exposure["bright_background_ratio"],
        overexposed_ratio=exposure["overexposed_ratio"],
        underexposed_ratio=exposure["underexposed_ratio"],
        exposure_level=interpret_exposure(
            exposure["mean_brightness"],
            exposure["overexposed_ratio"],
            exposure["underexposed_ratio"],
        ),

        # Контраст
        contrast_median=contrast_median,
        contrast_level=interpret_contrast(contrast_median),

        # Flicker
        flicker_mean=flicker["flicker_mean"],
        flicker_level=interpret_flicker(flicker["flicker_mean"]),

        # Стабильность
        motion_mean=stability["motion_mean"],
        motion_std=stability["motion_std"],
        stability_level=interpret_stability(stability["motion_mean"]),

        # Консистентность
        blur_std=consistency["blur_std"],
        brisque_std=consistency["brisque_std"],
        dover_technical_std=consistency.get("dover_technical_std"),
        brightness_std=consistency["brightness_std"],
        consistency_level=interpret_consistency(consistency["blur_rel_std"]),

        # Спикер
        face_presence_ratio=face["face_presence_ratio"],
        face_size_median=face["face_size_median"],

        # Доска
        board_detected=board_cv["board_detected"],
        board_glare_ratio=board_cv["board_glare_ratio"],
        board_contrast=board_cv["board_contrast"],
        board_glare_level=board_cv["board_glare_level"],
        board_readability_level=board_vlm["board_readability_level"],
        board_readability_score=board_vlm["board_readability_score"],
        board_surface_type=board_vlm["board_surface_type"],
        board_main_issues=board_vlm["board_main_issues"],
    )

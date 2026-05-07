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
    board_readability_level: Optional[str]  # "readable" / "marginal" / "unreadable"


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
        "--format", "bestvideo[vcodec^=avc1][ext=mp4]/bestvideo[ext=mp4]/bestvideo",
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
    """
    Технические параметры видео напрямую из метаданных контейнера.
    Без обработки кадров.
    """
    cmd = [
        "ffprobe", "-v", "quiet",
        "-print_format", "json",
        "-show_streams",
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

    # avg_frame_rate приходит как строка "30/1" → eval даёт float
    fps_raw = video_stream.get("avg_frame_rate", "0/1")
    try:
        fps = eval(fps_raw)  # noqa: S307
    except Exception:
        fps = 0.0

    meta = {
        "width":        int(video_stream["width"]),
        "height":       int(video_stream["height"]),
        "fps":          round(float(fps), 2),
        "bitrate_kbps": int(video_stream.get("bit_rate", 0)) // 1000,
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
    """Яркость, пересвет и недосвет по V-каналу HSV."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    v   = hsv[:, :, 2].astype(float) / 255.0
    return {
        "mean_brightness":    float(np.mean(v)),
        "overexposed_ratio":  float(np.mean(v > 0.95)),
        "underexposed_ratio": float(np.mean(v < 0.05)),
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
    if overexposed > OVEREXPOSED_RATIO_OK * 5:   # > 5%
        return "overexposed"
    if underexposed > UNDEREXPOSED_RATIO_OK * 4:  # > 20%
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

def detect_board_region(frame: np.ndarray) -> Optional[np.ndarray]:
    """
    Ищет наибольший прямоугольный контур — доску или экран.
    Возвращает бинарную маску или None если не найден.
    """
    gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    edges   = cv2.Canny(blurred, 50, 150)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    frame_area = frame.shape[0] * frame.shape[1]
    best_mask  = None
    best_area  = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < frame_area * 0.05 or area > frame_area * 0.90:
            continue

        peri   = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

        if len(approx) == 4 and area > best_area:
            best_area = area
            mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            cv2.fillPoly(mask, [approx], 1)
            best_mask = mask.astype(bool)

    return best_mask


def compute_board_metrics(frames: list[np.ndarray]) -> dict:
    """
    Читаемость доски / экрана.
    Берём медиану по всем quality-кадрам где регион был найден.

    Два сигнала:
      glare_ratio — засветка (окно бьёт в доску)
      board_contrast — локальный контраст (есть ли контент)

    Интерпретация пары:
      glare высокий + contrast низкий  → окно бьёт в доску
      glare низкий  + contrast низкий  → доска пустая или бледная
      glare низкий  + contrast высокий → всё хорошо
    """
    logger.info("Анализируем читаемость доски/экрана...")

    glare_scores    = []
    contrast_scores = []
    detected_count  = 0

    for frame in frames:
        mask = detect_board_region(frame)
        if mask is None:
            continue

        detected_count += 1

        # Засветка
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        v   = hsv[:, :, 2].astype(float) / 255.0
        glare = float(np.mean(v[mask] > 0.92))
        glare_scores.append(glare)

        # Локальный контраст
        gray   = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(float) / 255.0
        contrast = float(np.std(gray[mask]))
        contrast_scores.append(contrast)

    board_detected = detected_count > 0

    if not board_detected:
        logger.info("Доска/экран не обнаружены (скринкаст или кадр без явной доски)")
        return {
            "board_detected":         False,
            "board_glare_ratio":      None,
            "board_contrast":         None,
            "board_glare_level":      None,
            "board_readability_level": None,
        }

    glare_median    = round(float(np.median(glare_scores)), 4)
    contrast_median = round(float(np.median(contrast_scores)), 4)

    # Засветка
    if glare_median >= BOARD_GLARE_SEVERE:
        glare_level = "severe"
    elif glare_median >= BOARD_GLARE_PARTIAL:
        glare_level = "partial"
    else:
        glare_level = "ok"

    # Читаемость
    if contrast_median >= BOARD_CONTRAST_READABLE:
        readability = "readable"
    elif contrast_median >= BOARD_CONTRAST_MARGINAL:
        readability = "marginal"
    else:
        readability = "unreadable"

    logger.info(
        f"Доска: detected={detected_count}/{len(frames)} кадров, "
        f"glare={glare_median:.4f} ({glare_level}), "
        f"contrast={contrast_median:.4f} ({readability})"
    )

    return {
        "board_detected":         True,
        "board_glare_ratio":      glare_median,
        "board_contrast":         contrast_median,
        "board_glare_level":      glare_level,
        "board_readability_level": readability,
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
    board = compute_board_metrics(quality_frames)

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
        board_detected=board["board_detected"],
        board_glare_ratio=board["board_glare_ratio"],
        board_contrast=board["board_contrast"],
        board_glare_level=board["board_glare_level"],
        board_readability_level=board["board_readability_level"],
    )

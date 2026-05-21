"""
Флаг «Эмоция» — эмоциональный тон лектора.

Компоненты:
  1. audeering/wav2vec2 — arousal по 30-секундным окнам
  2. MediaPipe Pose — центроид тела (скорость и диапазон перемещения)
  3. Просодические контекстные метрики (F0_std, RMS_std, WPM) по окнам
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field, asdict

import numpy as np
import torch

logger = logging.getLogger(__name__)

# ─── Thresholds (откалиброваны на реальных лекционных видео) ──────────────────

AROUSAL_HIGH_MEAN     = 0.75   # слишком театрально
AROUSAL_FLAT_MEAN     = 0.30   # слишком плоско
AROUSAL_FLAT_STD      = 0.05   # механическая/монотонная подача
AROUSAL_VOLATILE_STD  = 0.15   # эмоциональные качели

CHUNK_SEC             = 30.0   # размер окна в секундах
SAMPLE_RATE           = 16_000

# ─── Data structures ──────────────────────────────────────────────────────────

@dataclass
class EmotionArousalResult:
    mean: float
    std: float
    min: float
    max: float
    n_chunks: int
    flat_flag: bool
    high_flag: bool
    volatile_flag: bool
    triggered_by: list[str] = field(default_factory=list)


@dataclass
class EmotionPoseResult:
    available: bool
    velocity_mean: float = 0.0
    position_range: float = 0.0
    error: str | None = None


@dataclass
class EmotionFlagResult:
    flag: bool
    confidence: float
    arousal: dict
    prosodics: dict
    pose: dict
    triggered_by: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


# ─── Arousal model ────────────────────────────────────────────────────────────

def _load_emotion_model(device: str):
    """Загружаем audeering модель через кастомный класс."""
    from transformers import Wav2Vec2Processor

    # Импортируем кастомный класс (файл должен быть в pipeline/)
    try:
        from emotion_model_audeering import EmotionModel
    except ImportError:
        raise ImportError(
            "Не найден emotion_model_audeering.py. "
            "Убедитесь что файл лежит в папке pipeline/."
        )

    model_id = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"
    processor = Wav2Vec2Processor.from_pretrained(model_id)
    model = EmotionModel.from_pretrained(model_id).to(device)
    model.eval()
    return processor, model


def _predict_arousal_chunks(
    audio_path: str,
    processor,
    model,
    device: str,
) -> np.ndarray:
    """
    Разбиваем аудио на окна и считаем arousal для каждого.
    Возвращает массив [arousal_per_chunk].
    """
    import soundfile as sf

    signal, sr = sf.read(audio_path, dtype="float32")
    if signal.ndim > 1:
        signal = signal.mean(axis=1)

    # Ресемплируем если нужно
    if sr != SAMPLE_RATE:
        import librosa
        signal = librosa.resample(signal, orig_sr=sr, target_sr=SAMPLE_RATE)

    chunk_len = int(CHUNK_SEC * SAMPLE_RATE)
    arousal_values = []

    for start in range(0, len(signal), chunk_len):
        chunk = signal[start : start + chunk_len]
        if len(chunk) < SAMPLE_RATE:  # меньше 1 сек — пропускаем
            continue

        inputs = processor(chunk, sampling_rate=SAMPLE_RATE, return_tensors="pt", padding=True)
        input_values = inputs.input_values.to(device)

        with torch.no_grad():
            _, logits = model(input_values)

        # logits: [valence, arousal, dominance]
        arousal = logits[0][1].item()
        arousal_values.append(arousal)

    return np.array(arousal_values)


def _run_arousal(audio_path: str, device: str = "cpu") -> EmotionArousalResult:
    """Запускаем arousal-анализ."""
    try:
        processor, model = _load_emotion_model(device)
        arousal_arr = _predict_arousal_chunks(audio_path, processor, model, device)
    except Exception as e:
        logger.error(f"Ошибка arousal модели: {e}")
        return EmotionArousalResult(
            mean=0.0, std=0.0, min=0.0, max=0.0, n_chunks=0,
            flat_flag=False, high_flag=False, volatile_flag=False,
            triggered_by=[f"error: {e}"],
        )

    if len(arousal_arr) == 0:
        return EmotionArousalResult(
            mean=0.0, std=0.0, min=0.0, max=0.0, n_chunks=0,
            flat_flag=False, high_flag=False, volatile_flag=False,
            triggered_by=["no_chunks"],
        )

    mean = float(np.mean(arousal_arr))
    std  = float(np.std(arousal_arr))
    mn   = float(np.min(arousal_arr))
    mx   = float(np.max(arousal_arr))

    triggered = []
    flat     = std < AROUSAL_FLAT_STD or mean < AROUSAL_FLAT_MEAN
    high     = mean > AROUSAL_HIGH_MEAN
    volatile = std > AROUSAL_VOLATILE_STD

    if flat:     triggered.append(f"flat (mean={mean:.3f}, std={std:.3f})")
    if high:     triggered.append(f"high_arousal (mean={mean:.3f})")
    if volatile: triggered.append(f"volatile (std={std:.3f})")

    return EmotionArousalResult(
        mean=round(mean, 4),
        std=round(std, 4),
        min=round(mn, 4),
        max=round(mx, 4),
        n_chunks=len(arousal_arr),
        flat_flag=flat,
        high_flag=high,
        volatile_flag=volatile,
        triggered_by=triggered,
    )


# ─── MediaPipe Pose (центроид) ────────────────────────────────────────────────

def _run_pose(frame_paths: list[str], face_size_median: float = 0.1) -> EmotionPoseResult:
    """
    Трекаем центроид плеч лектора по кадрам.
    Возвращает velocity_mean и position_range.
    """
    if not frame_paths:
        return EmotionPoseResult(available=False, error="no_frames")

    # Крупный план — Pose не поможет
    if face_size_median > 0.15:
        return EmotionPoseResult(available=False, error="tight_shot_face_size_too_large")

    try:
        import mediapipe as mp
        import cv2
    except ImportError:
        return EmotionPoseResult(available=False, error="mediapipe_not_installed")

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

    centroids = []

    for path in frame_paths:
        img = cv2.imread(path)
        if img is None:
            continue
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = pose.process(rgb)
        if not result.pose_landmarks:
            continue

        lm = result.pose_landmarks.landmark
        left_shoulder  = lm[mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER]

        cx = (left_shoulder.x + right_shoulder.x) / 2
        cy = (left_shoulder.y + right_shoulder.y) / 2
        centroids.append((cx, cy))

    pose.close()

    if len(centroids) < 2:
        return EmotionPoseResult(available=False, error="insufficient_detections")

    positions = np.array(centroids)
    diffs = np.sqrt(np.diff(positions[:, 0])**2 + np.diff(positions[:, 1])**2)
    velocity_mean  = float(np.mean(diffs))
    position_range = float(np.max(positions[:, 0]) - np.min(positions[:, 0]))

    return EmotionPoseResult(
        available=True,
        velocity_mean=round(velocity_mean, 4),
        position_range=round(position_range, 4),
    )


# ─── Main entry point ─────────────────────────────────────────────────────────

def run(
    audio_path: str,
    device: str = "cpu",
    frame_paths: list[str] | None = None,
    face_size_median: float = 0.1,
    prosodics: dict | None = None,
) -> EmotionFlagResult:
    """
    Запускает флаг Эмоция.

    Args:
        audio_path: путь к WAV 16kHz
        device: 'cpu' или 'cuda'
        frame_paths: кадры для MediaPipe Pose (опционально)
        face_size_median: медианный размер лица из video_quality (для адаптации)
        prosodics: словарь с F0_std, RMS_std, WPM (контекст, не флаг)

    Returns:
        EmotionFlagResult
    """

    logger.info("── Флаг Эмоция: arousal по окнам ──")
    arousal = _run_arousal(audio_path, device)

    pose = EmotionPoseResult(available=False)
    if frame_paths:
        logger.info("── Флаг Эмоция: MediaPipe Pose ──")
        pose = _run_pose(frame_paths, face_size_median)

    triggered = list(arousal.triggered_by)
    flag = arousal.flat_flag or arousal.high_flag or arousal.volatile_flag

    # Уверенность пропорциональна отклонению от нормы
    if arousal.n_chunks == 0:
        confidence = 0.0
    else:
        deviation = max(
            abs(arousal.mean - 0.5) / 0.25,
            abs(arousal.std - 0.08) / 0.07,
        )
        confidence = round(min(deviation, 1.0), 3)

    return EmotionFlagResult(
        flag=flag,
        confidence=confidence,
        arousal={
            "mean":         arousal.mean,
            "std":          arousal.std,
            "min":          arousal.min,
            "max":          arousal.max,
            "n_chunks":     arousal.n_chunks,
            "flat_flag":    arousal.flat_flag,
            "high_flag":    arousal.high_flag,
            "volatile_flag": arousal.volatile_flag,
            "thresholds": {
                "high_mean":    AROUSAL_HIGH_MEAN,
                "flat_mean":    AROUSAL_FLAT_MEAN,
                "flat_std":     AROUSAL_FLAT_STD,
                "volatile_std": AROUSAL_VOLATILE_STD,
            },
        },
        prosodics=prosodics or {},
        pose={
            "available":      pose.available,
            "velocity_mean":  pose.velocity_mean,
            "position_range": pose.position_range,
            "error":          pose.error,
        },
        triggered_by=triggered,
    )

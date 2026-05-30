"""
Флаг «Эмоция» — эмоциональный тон лектора.
 
Компоненты:
  1. audeering/wav2vec2 — arousal по 30-секундным окнам
  2. MediaPipe Pose — движение лектора (центроид + жестикуляция запястий)
  3. Просодические контекстные метрики (F0_std, RMS_std, WPM) по окнам
"""
 
from __future__ import annotations
 
import logging
import os
from dataclasses import dataclass, field, asdict
 
import numpy as np
import torch
 
logger = logging.getLogger(__name__)
 
# ─── Arousal thresholds ───────────────────────────────────────────────────────
 
AROUSAL_HIGH_MEAN    = 0.75
AROUSAL_FLAT_MEAN    = 0.30
AROUSAL_FLAT_STD     = 0.04
AROUSAL_VOLATILE_STD = 0.15
CHUNK_SEC            = 30.0
SAMPLE_RATE          = 16_000
 
# ─── Movement thresholds ──────────────────────────────────────────────────────
 
POSE_LANDMARKER_MODEL_PATH = "/tmp/pose_landmarker_lite.task"
 
# Центроид: выборка 1 кадр каждые N секунд по всему видео
CENTROID_SAMPLE_SEC  = 3.0
 
# Запястья: плотные окна для жестикуляции
WRIST_WINDOW_SEC     = 15    # длина окна
WRIST_WINDOW_FPS     = 5     # кадров в секунду внутри окна
WRIST_AMPLITUDE_THR  = 0.25  # размах > 25% ширины кадра = жестикуляция
WRIST_VELOCITY_THR   = 0.015 # минимальная скорость
 
 
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
class EmotionFlagResult:
    flag: bool
    confidence: float
    arousal: dict
    prosodics: dict
    movement: dict
    triggered_by: list[str] = field(default_factory=list)
 
    def to_dict(self) -> dict:
        return asdict(self)
 
 
# ─── Arousal model ────────────────────────────────────────────────────────────
 
def _load_emotion_model(device: str):
    from transformers import Wav2Vec2Processor
    try:
        from emotion_model_audeering import EmotionModel
    except ImportError as e:
        raise ImportError(
            f"Не удалось импортировать EmotionModel: {e}. "
            "Убедитесь что emotion_model_audeering.py лежит в pipeline/."
        ) from e
 
    model_id = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"
    processor = Wav2Vec2Processor.from_pretrained(model_id)
    model = EmotionModel.from_pretrained(model_id).to(device)
    model.eval()
    return processor, model
 
 
def _predict_arousal_chunks(audio_path, processor, model, device) -> np.ndarray:
    import soundfile as sf
    signal, sr = sf.read(audio_path, dtype="float32")
    if signal.ndim > 1:
        signal = signal.mean(axis=1)
    if sr != SAMPLE_RATE:
        import librosa
        signal = librosa.resample(signal, orig_sr=sr, target_sr=SAMPLE_RATE)
 
    chunk_len = int(CHUNK_SEC * SAMPLE_RATE)
    arousal_values = []
    for start in range(0, len(signal), chunk_len):
        chunk = signal[start: start + chunk_len]
        if len(chunk) < SAMPLE_RATE:
            continue
        inputs = processor(chunk, sampling_rate=SAMPLE_RATE,
                           return_tensors="pt", padding=True)
        with torch.no_grad():
            _, logits = model(inputs.input_values.to(device))
        arousal_values.append(logits[0][1].item())
    return np.array(arousal_values)
 
 
def _run_arousal(audio_path: str, device: str = "cpu") -> EmotionArousalResult:
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
    flat     = std < AROUSAL_FLAT_STD or mean < AROUSAL_FLAT_MEAN
    high     = mean > AROUSAL_HIGH_MEAN
    volatile = std > AROUSAL_VOLATILE_STD
 
    triggered = []
    if flat:     triggered.append(f"flat (mean={mean:.3f}, std={std:.3f})")
    if high:     triggered.append(f"high_arousal (mean={mean:.3f})")
    if volatile: triggered.append(f"volatile (std={std:.3f})")
 
    return EmotionArousalResult(
        mean=round(mean, 4), std=round(std, 4),
        min=round(float(np.min(arousal_arr)), 4),
        max=round(float(np.max(arousal_arr)), 4),
        n_chunks=len(arousal_arr),
        flat_flag=bool(flat), high_flag=bool(high), volatile_flag=bool(volatile),
        triggered_by=triggered,
    )
 
 
# ─── Movement analysis (centroid + wrist gesture) ─────────────────────────────
 
def _run_movement_analysis(video_path: str) -> dict:
    """
    Единый анализ движений лектора из одного video_path:
 
    Centroid (редкая выборка — 1 кадр каждые CENTROID_SAMPLE_SEC):
      - position_range: диапазон горизонтального перемещения лектора
      - velocity_mean:  средняя скорость перемещения центроида плеч
 
    Запястья (плотные окна — WRIST_WINDOW_FPS fps):
      - amplitude_mean: средний размах жестикуляции по окнам
      - gesture_active: превышает ли амплитуда порог
    """
    import cv2
 
    try:
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision
    except ImportError:
        return {"available": False, "error": "mediapipe_not_installed"}
 
    if not os.path.exists(POSE_LANDMARKER_MODEL_PATH):
        return {"available": False,
                "error": f"model not found at {POSE_LANDMARKER_MODEL_PATH}"}
 
    try:
        base_options = mp_python.BaseOptions(
            model_asset_path=POSE_LANDMARKER_MODEL_PATH)
        options = mp_vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.IMAGE,
        )
        landmarker = mp_vision.PoseLandmarker.create_from_options(options)
    except Exception as e:
        return {"available": False, "error": f"landmarker_init: {e}"}
 
    cap = cv2.VideoCapture(video_path)
    fps_native   = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / fps_native
 
    # Пропускаем первые и последние 5%
    start_frame = int(total_frames * 0.05)
    end_frame   = int(total_frames * 0.95)
 
    def detect(frame_bgr):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        res = landmarker.detect(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
        if res.pose_landmarks:
            return res.pose_landmarks[0]
        return None
 
    # ── Centroid: редкая выборка по всему видео ──────────────────────────────
    centroid_step = max(1, int(CENTROID_SAMPLE_SEC * fps_native))
    centroids = []
 
    for frame_idx in range(start_frame, end_frame, centroid_step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue
        lms = detect(frame)
        if lms:
            cx = (lms[11].x + lms[12].x) / 2
            cy = (lms[11].y + lms[12].y) / 2
            centroids.append((cx, cy))
 
    centroid_result = {"available": False}
    if len(centroids) >= 2:
        pos = np.array(centroids)
        diffs = np.sqrt(np.diff(pos[:, 0])**2 + np.diff(pos[:, 1])**2)
        centroid_result = {
            "available":      True,
            "velocity_mean":  round(float(np.mean(diffs)), 4),
            "position_range": round(float(pos[:, 0].max() - pos[:, 0].min()), 4),
            "n_frames":       len(centroids),
        }
 
    # ── Запястья: плотные окна ────────────────────────────────────────────────
    # Динамическое кол-во окон: 1 на каждые 10 мин, минимум 3, максимум 8
    n_windows  = max(3, min(8, int(duration_sec / 600)))
    win_frames = int(WRIST_WINDOW_SEC * fps_native)
    win_step   = max(1, int(fps_native / WRIST_WINDOW_FPS))
 
    window_starts = np.linspace(
        start_frame,
        max(start_frame, end_frame - win_frames),
        n_windows, dtype=int,
    )
 
    window_amplitudes = []
    window_velocities = []
 
    for w_start in window_starts:
        wrist_pos = []
        for frame_idx in range(int(w_start), int(w_start) + win_frames, win_step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                continue
            lms = detect(frame)
            if lms:
                wrist_pos.append((lms[15].x, lms[15].y,   # left wrist
                                   lms[16].x, lms[16].y))  # right wrist
 
        if len(wrist_pos) < 3:
            continue
 
        pos = np.array(wrist_pos)
        # Амплитуда: максимальный размах по обоим запястьям
        amp_l = float(np.sqrt(
            (pos[:, 0].max() - pos[:, 0].min())**2 +
            (pos[:, 1].max() - pos[:, 1].min())**2
        ))
        amp_r = float(np.sqrt(
            (pos[:, 2].max() - pos[:, 2].min())**2 +
            (pos[:, 3].max() - pos[:, 3].min())**2
        ))
        window_amplitudes.append(max(amp_l, amp_r))
 
        # Скорость: среднее смещение между кадрами
        diffs_l = np.sqrt(np.diff(pos[:, 0])**2 + np.diff(pos[:, 1])**2)
        diffs_r = np.sqrt(np.diff(pos[:, 2])**2 + np.diff(pos[:, 3])**2)
        window_velocities.append(float(np.mean(np.maximum(diffs_l, diffs_r))))
 
    cap.release()
    landmarker.close()
 
    wrist_result = {"available": False}
    if window_amplitudes:
        amp_mean = float(np.mean(window_amplitudes))
        vel_mean = float(np.mean(window_velocities))

        centroid_range = centroid_result.get("position_range", 0.0) if centroid_result.get("available") else 0.0
        gesture_ratio  = amp_mean / (0.1 + centroid_range)
        gesture_active = gesture_ratio > 1.35

        wrist_result = {
            "available":      True,
            "amplitude_mean": round(amp_mean, 4),
            "velocity_mean":  round(vel_mean, 4),
            "gesture_ratio":  round(gesture_ratio, 4),
            "n_windows":      len(window_amplitudes),
            "gesture_active": bool(gesture_active),
        }
 
    return {
        "available": centroid_result["available"] or wrist_result["available"],
        "centroid":  centroid_result,
        "wrist":     wrist_result,
    }
 
 
# ─── Main entry point ─────────────────────────────────────────────────────────
 
def run(
    audio_path: str,
    device: str = "cpu",
    video_path: str | None = None,
    prosodics: dict | None = None,
) -> EmotionFlagResult:
    """
    Запускает флаг Эмоция.
 
    Args:
        audio_path: путь к WAV 16kHz
        device:     'cpu' или 'cuda'
        video_path: путь к видео для MediaPipe (опционально)
        prosodics:  F0_std, RMS_std, WPM по окнам (контекст)
    """
 
    logger.info("── Флаг Эмоция: arousal по окнам ──")
    arousal = _run_arousal(audio_path, device)
 
    movement = {"available": False}
    if video_path:
        logger.info("── Флаг Эмоция: анализ движений (centroid + запястья) ──")
        movement = _run_movement_analysis(video_path)
 
    triggered = list(arousal.triggered_by)

    gesture_active = movement.get("wrist", {}).get("gesture_active", False)

    flag = arousal.flat_flag or arousal.high_flag or arousal.volatile_flag or gesture_active

    if gesture_active:
        triggered.append("gesture_active")
 
    if arousal.n_chunks == 0:
        confidence = 0.0
    else:
        deviation = max(
            abs(arousal.mean - 0.5) / 0.25,
            abs(arousal.std - 0.08) / 0.07,
        )
        confidence = round(min(float(deviation), 1.0), 3)
 
    return EmotionFlagResult(
        flag=flag,
        confidence=confidence,
        arousal={
            "mean":          arousal.mean,
            "std":           arousal.std,
            "min":           arousal.min,
            "max":           arousal.max,
            "n_chunks":      arousal.n_chunks,
            "flat_flag":     arousal.flat_flag,
            "high_flag":     arousal.high_flag,
            "volatile_flag": arousal.volatile_flag,
            "thresholds": {
                "high_mean":    AROUSAL_HIGH_MEAN,
                "flat_mean":    AROUSAL_FLAT_MEAN,
                "flat_std":     AROUSAL_FLAT_STD,
                "volatile_std": AROUSAL_VOLATILE_STD,
            },
        },
        prosodics=prosodics or {},
        movement=movement,
        triggered_by=triggered,
    )
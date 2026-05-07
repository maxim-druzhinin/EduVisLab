import torch

# ─── Device auto-detection ───────────────────────────────────────────────────

if torch.cuda.is_available():
    DEVICE = "cuda"
    COMPUTE_TYPE = "float16"
else:
    # faster-whisper не поддерживает MPS
    DEVICE = "cpu"
    COMPUTE_TYPE = "int8"

# ─── Transcription ───────────────────────────────────────────────────────────

WHISPER_MODEL = "dropbox-dash/faster-whisper-large-v3-turbo" # large-v2?
WHISPER_LANGUAGE = "ru"
WHISPER_CONDITION_ON_PREV = False   # сохраняем filler words

# ─── Audio quality thresholds ────────────────────────────────────────────────

# DNSMOS (1–5 шкала, ITU-T P.835 / Reddy et al. ICASSP 2022)
MOS_GOOD = 3.5
MOS_BAD  = 2.5

# LUFS (ITU-R BS.1770-5 / EBU R128 / AES TD1004)
LUFS_NORM_HIGH = -14.0   # YouTube не режет ниже этого
LUFS_NORM_LOW  = -23.0   # EBU R128 нижняя граница нормы
LUFS_TOO_QUIET = -30.0   # при этом YouTube не поднимает сам
LUFS_TOO_LOUD  =  -6.0   # граничит с клиппингом

# Клиппинг (эмпирический стандарт аудиоинженерии)
# Клиппинг — три уровня
# clipping_ratio (прямой метод — амплитуда > 0.99)
CLIPPING_RATIO_MILD   = 0.0001   # 0.01% сэмплов
CLIPPING_RATIO_SEVERE = 0.001    # 0.1% сэмплов
CLIPPING_THRESHOLD      = 0.99   # амплитуда выше которой считаем клиппинг

# crest_factor (пик / RMS — низкий = срезанные пики)
# для чистого голоса обычно 10–20 dB
CREST_FACTOR_MILD   = 10.0   # ниже → подозрение
CREST_FACTOR_SEVERE =  6.0   # ниже → явный клиппинг

# flattened_peaks_ratio (доля плоских участков на пиках)
FLAT_PEAKS_MILD   = 0.001   # 0.1%
FLAT_PEAKS_SEVERE = 0.005   # 0.5%

# SNR (Loizou 2007 "Speech Enhancement" / ANSI S3.5-1997)
SNR_GOOD = 20.0   # > 20 dB — чистая речь
SNR_OK   = 10.0   # 10–20 dB — терпимо
# < 10 dB — шум мешает пониманию

# ─── silero-VAD ───────────────────────────────────────────────────────────────

VAD_THRESHOLD          = 0.5    # порог вероятности речи
VAD_MIN_SPEECH_MS      = 250    # минимальная длина речевого сегмента (мс)
VAD_MIN_SILENCE_MS     = 100    # минимальная пауза между сегментами (мс)
VAD_SPEECH_PAD_MS      = 30     # отступ вокруг речевых сегментов (мс)

# ─── Флаг Унылость ───────────────────────────────────────────────────────────
 
# Итоговый порог (0..1) — выше = уныло
# Подбирается на тестовых видео, стартовое значение
DULLNESS_THRESHOLD = 0.5
 
# Веса акустики и лингвистики в итоговом score
# Акустика надёжнее — подкреплена Shi et al.
DULLNESS_WEIGHT_ACOUSTIC   = 0.6
DULLNESS_WEIGHT_LINGUISTIC = 0.4
 
# WPM (Toastmasters / ASHA)
WPM_TOO_SLOW = 100   # ниже → усыпляет
WPM_TOO_FAST = 180   # выше → снижается понимание
 
# Паузы-хезитации (Goldman-Eisler 1968, Levelt 1989)
HESITATION_PAUSE_THRESHOLD = 0.5   # секунды

# ─── Paths ───────────────────────────────────────────────────────────────────

AUDIO_DIR  = "/tmp/pipeline_audio"   # временные аудиофайлы
OUTPUT_DIR = "/tmp/pipeline_output"  # результаты


# ─── Video quality thresholds ────────────────────────────────────────────────
# Добавить в config.py после существующих порогов

# Разрешение
RESOLUTION_HIGH   = 1080   # px по высоте
RESOLUTION_NORMAL = 720
RESOLUTION_LOW    = 480
# < 480 → bad

# DOVER (перцептивное качество, 0..1)
DOVER_GOOD = 0.55
DOVER_BAD  = 0.35

# BRISQUE (артефакты/зерно, 0 = идеально, 100 = плохо)
BRISQUE_GOOD = 20
BRISQUE_BAD  = 50

# Blur — дисперсия Лапласиана (при 1080p)
BLUR_SHARP  = 100   # > sharp
BLUR_SOFT   = 50    # > soft, иначе blurry

# Экспозиция (V-канал HSV, 0..1)
BRIGHTNESS_MIN        = 0.15   # ниже → темно
BRIGHTNESS_MAX        = 0.85   # выше → пересвет
OVEREXPOSED_RATIO_OK  = 0.01   # > 1% → проблема
UNDEREXPOSED_RATIO_OK = 0.05   # > 5% → проблема

# Контраст (std яркости)
CONTRAST_GOOD = 0.15
CONTRAST_OK   = 0.08
# < 0.08 → low

# Мерцание (средняя разница яркости между кадрами, 0–255)
FLICKER_MILD   = 2
FLICKER_SEVERE = 6

# Стабильность — vidstabdetect (пиксели при 1080p)
# (640px, median по кадрам):
STABILITY_OK        = 0.3
STABILITY_SHAKY     = 1.0
STABILITY_VERY_SHAKY = 3.0

# Временная консистентность (relative std = std / mean)
CONSISTENCY_MODERATE     = 0.15
CONSISTENCY_INCONSISTENT = 0.35

# Видимость спикера
FACE_PRESENCE_HIGH = 0.8   # > постоянно в кадре
FACE_PRESENCE_LOW  = 0.4   # < скринкаст или за кадром
FACE_SIZE_OK       = 0.05  # > 5% площади кадра — нормально
FACE_SIZE_SMALL    = 0.02  # > 2% — далеко, < 2% — слишком далеко

# Читаемость доски
BOARD_GLARE_PARTIAL = 0.05   # > 5% засвечено
BOARD_GLARE_SEVERE  = 0.20   # > 20% засвечено
BOARD_CONTRAST_READABLE  = 0.12   # > readable
BOARD_CONTRAST_MARGINAL  = 0.06   # > marginal, иначе unreadable

# Пути для видео
VIDEO_DIR = "/tmp/pipeline_video"   # скачанные видеофайлы
FRAMES_DIR = "/tmp/pipeline_frames" # извлечённые кадры

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
STABILITY_OK      = 0.5
STABILITY_SHAKY   = 2.0
STABILITY_VERY_SHAKY = 6.0

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

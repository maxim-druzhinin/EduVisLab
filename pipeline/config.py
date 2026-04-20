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

# ─── Paths ───────────────────────────────────────────────────────────────────

AUDIO_DIR  = "/tmp/pipeline_audio"   # временные аудиофайлы
OUTPUT_DIR = "/tmp/pipeline_output"  # результаты

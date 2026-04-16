"""
Модуль технического качества звука.

Метрики:
  - DNSMOS (ovrl_mos, sig_mos, bak_mos) — через speechmos
  - LUFS  — через pyloudnorm
  - Клиппинг — прямой расчёт numpy
  - SNR — через silero-VAD сегменты + numpy RMS

Все метрики считаются по WAV 16kHz моно
(ffmpeg-конвертация делается в модуле transcription.py).
"""

import logging
from dataclasses import dataclass

import numpy as np
import torch
import soundfile as sf
import pyloudnorm as pyln
from speechmos import dnsmos

from config import (
    MOS_GOOD, MOS_BAD,
    LUFS_NORM_HIGH, LUFS_NORM_LOW, LUFS_TOO_QUIET, LUFS_TOO_LOUD,
    CLIPPING_THRESHOLD, CLIPPING_RATIO_CRITICAL,
    SNR_GOOD, SNR_OK,
    VAD_THRESHOLD, VAD_MIN_SPEECH_MS, VAD_MIN_SILENCE_MS, VAD_SPEECH_PAD_MS,
)

logger = logging.getLogger(__name__)


# ─── Data classes ─────────────────────────────────────────────────────────────

@dataclass
class VADSegment:
    start: float   # секунды
    end: float     # секунды

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class AudioQualityResult:
    # DNSMOS
    ovrl_mos: float
    sig_mos: float
    bak_mos: float

    # LUFS
    lufs: float

    # Клиппинг
    clipping_ratio: float
    clipping_detected: bool

    # SNR
    snr_db: float

    # Интерпретации
    mos_quality: str      # "good" / "ok" / "bad"
    lufs_quality: str     # "normal" / "quiet" / "loud" / "too_quiet" / "too_loud"
    snr_quality: str      # "good" / "ok" / "bad"

    # VAD сегменты — переиспользуются модулем Унылость
    speech_segments: list[VADSegment]


# ─── silero-VAD ───────────────────────────────────────────────────────────────

def get_vad_segments(audio: np.ndarray, sr: int) -> list[VADSegment]:
    """
    Запускает silero-VAD и возвращает список речевых сегментов.
    """
    logger.info("Запускаем silero-VAD...")

    # Загружаем модель из torch.hub (кэшируется после первого запуска)
    model, utils = torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        force_reload=False,
        trust_repo=True,
    )
    get_speech_timestamps = utils[0]

    # silero-VAD принимает torch tensor
    audio_tensor = torch.FloatTensor(audio)

    speech_timestamps = get_speech_timestamps(
        audio_tensor,
        model,
        threshold=VAD_THRESHOLD,
        sampling_rate=sr,
        min_speech_duration_ms=VAD_MIN_SPEECH_MS,
        min_silence_duration_ms=VAD_MIN_SILENCE_MS,
        speech_pad_ms=VAD_SPEECH_PAD_MS,
        return_seconds=True,  # возвращаем в секундах, не сэмплах
    )

    segments = [
        VADSegment(start=ts["start"], end=ts["end"])
        for ts in speech_timestamps
    ]

    total_speech = sum(s.duration for s in segments)
    logger.info(
        f"VAD: {len(segments)} речевых сегментов, "
        f"{total_speech:.1f}с речи из {len(audio)/sr:.1f}с всего"
    )
    return segments


# ─── DNSMOS ───────────────────────────────────────────────────────────────────

def compute_dnsmos(audio: np.ndarray, sr: int) -> dict:
    """
    Оценка качества звука через DNSMOS P.835 (Microsoft).
    Без референса — идеально для YouTube.

    Источник: Reddy et al. ICASSP 2022, arXiv:2110.01763
    """
    logger.info("Считаем DNSMOS...")
    result = dnsmos.run(audio, sr=sr)

    scores = {
        "ovrl_mos": round(float(result["ovrl_mos"]), 3),
        "sig_mos":  round(float(result["sig_mos"]),  3),
        "bak_mos":  round(float(result["bak_mos"]),  3),
    }
    logger.info(f"DNSMOS: {scores}")
    return scores


def interpret_mos(ovrl_mos: float) -> str:
    if ovrl_mos >= MOS_GOOD:
        return "good"
    elif ovrl_mos >= MOS_BAD:
        return "ok"
    else:
        return "bad"


# ─── LUFS ─────────────────────────────────────────────────────────────────────

def compute_lufs(audio: np.ndarray, sr: int) -> float:
    """
    Интегральная громкость по стандарту ITU-R BS.1770-5 / EBU R128.
    """
    logger.info("Считаем LUFS...")
    meter = pyln.Meter(sr)
    lufs = meter.integrated_loudness(audio.astype(np.float64))
    logger.info(f"LUFS: {lufs:.2f}")
    return round(float(lufs), 2)


def interpret_lufs(lufs: float) -> str:
    if lufs <= LUFS_TOO_QUIET:
        return "too_quiet"    # YouTube не поднимает — плохой UX
    elif lufs <= LUFS_NORM_LOW:
        return "quiet"        # тихо, но YouTube поднимет в плеере
    elif lufs <= LUFS_NORM_HIGH:
        return "normal"       # норма
    elif lufs <= LUFS_TOO_LOUD:
        return "loud"         # YouTube порежет
    else:
        return "too_loud"     # граничит с клиппингом


# ─── Clipping ─────────────────────────────────────────────────────────────────

def compute_clipping(audio: np.ndarray) -> tuple[float, bool]:
    """
    Доля сэмплов с амплитудой > 0.99.
    Клиппинг = перегруз микрофона → хруст и дисторшн.
    """
    ratio = float(np.mean(np.abs(audio) > CLIPPING_THRESHOLD))
    detected = ratio > CLIPPING_RATIO_CRITICAL
    logger.info(f"Клиппинг: ratio={ratio:.6f}, detected={detected}")
    return round(ratio, 6), detected


# ─── SNR ──────────────────────────────────────────────────────────────────────

def compute_snr(
    audio: np.ndarray,
    sr: int,
    speech_segments: list[VADSegment],
) -> float:
    """
    SNR через silero-VAD сегменты:
      - шум  = RMS во время пауз
      - сигнал = RMS во время речи

    SNR (dB) = 20 * log10(RMS_speech / RMS_noise)

    Источники: Loizou (2007) "Speech Enhancement", ANSI S3.5-1997
    """
    if not speech_segments:
        logger.warning("SNR: нет речевых сегментов, возвращаем 0")
        return 0.0

    # Собираем сэмплы речи и тишины
    speech_samples = []
    silence_samples = []

    # Сортируем сегменты по времени
    sorted_segs = sorted(speech_segments, key=lambda s: s.start)

    prev_end = 0.0
    for seg in sorted_segs:
        start_idx = int(seg.start * sr)
        end_idx   = int(seg.end * sr)
        prev_end_idx = int(prev_end * sr)

        # Тишина до этого сегмента
        if prev_end_idx < start_idx:
            silence_samples.append(audio[prev_end_idx:start_idx])

        # Речь этого сегмента
        speech_samples.append(audio[start_idx:end_idx])
        prev_end = seg.end

    # Тишина после последнего сегмента
    last_end_idx = int(sorted_segs[-1].end * sr)
    if last_end_idx < len(audio):
        silence_samples.append(audio[last_end_idx:])

    if not speech_samples or not silence_samples:
        logger.warning("SNR: недостаточно данных для расчёта")
        return 0.0

    speech_concat  = np.concatenate(speech_samples)
    silence_concat = np.concatenate(silence_samples)

    rms_speech  = np.sqrt(np.mean(speech_concat ** 2))
    rms_silence = np.sqrt(np.mean(silence_concat ** 2))

    if rms_silence < 1e-10:
        logger.info("SNR: тишина практически нулевая — очень чистый звук")
        return 99.0  # условно «идеально»

    snr = 20 * np.log10(rms_speech / rms_silence)
    logger.info(f"SNR: {snr:.1f} dB (RMS_speech={rms_speech:.4f}, RMS_noise={rms_silence:.4f})")
    return round(float(snr), 1)


def interpret_snr(snr_db: float) -> str:
    if snr_db >= SNR_GOOD:
        return "good"
    elif snr_db >= SNR_OK:
        return "ok"
    else:
        return "bad"


# ─── Main entry point ─────────────────────────────────────────────────────────

def run(wav_path: str) -> AudioQualityResult:
    """
    Полный анализ технического качества звука.
    Принимает путь к WAV 16kHz моно.
    """
    logger.info(f"Анализируем качество: {wav_path}")

    # Загружаем аудио
    audio, sr = sf.read(wav_path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)  # на случай если стерео проскочило

    assert sr == 16000, f"Ожидается 16kHz, получено {sr}Hz. Проверь ffmpeg-конвертацию."

    # 1. VAD — нужен для SNR и переиспользуется в модуле Унылость
    speech_segments = get_vad_segments(audio, sr)

    # 2. DNSMOS
    mos_scores = compute_dnsmos(audio, sr)

    # 3. LUFS
    lufs = compute_lufs(audio, sr)

    # 4. Клиппинг
    clipping_ratio, clipping_detected = compute_clipping(audio)

    # 5. SNR
    snr_db = compute_snr(audio, sr, speech_segments)

    return AudioQualityResult(
        ovrl_mos=mos_scores["ovrl_mos"],
        sig_mos=mos_scores["sig_mos"],
        bak_mos=mos_scores["bak_mos"],
        lufs=lufs,
        clipping_ratio=clipping_ratio,
        clipping_detected=clipping_detected,
        snr_db=snr_db,
        mos_quality=interpret_mos(mos_scores["ovrl_mos"]),
        lufs_quality=interpret_lufs(lufs),
        snr_quality=interpret_snr(snr_db),
        speech_segments=speech_segments,
    )

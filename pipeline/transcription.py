"""
Модуль транскрипции.

Шаги:
  1. yt-dlp  → скачать аудиодорожку с YouTube
  2. ffmpeg  → конвертировать в моно WAV 16kHz
  3. faster-whisper → транскрипция с segment-level timestamps
  4. WPM из длительности сегментов и числа слов
"""

import os
import subprocess
import logging
from pathlib import Path
from dataclasses import dataclass

from faster_whisper import WhisperModel

from config import (
    DEVICE, COMPUTE_TYPE,
    WHISPER_MODEL, WHISPER_LANGUAGE, WHISPER_CONDITION_ON_PREV,
    AUDIO_DIR,
)

logger = logging.getLogger(__name__)


# ─── Data classes ─────────────────────────────────────────────────────────────

@dataclass
class Segment:
    text: str
    start: float   # секунды
    end: float     # секунды

    @property
    def duration(self) -> float:
        return self.end - self.start

    @property
    def word_count(self) -> int:
        return len(self.text.strip().split())


@dataclass
class TranscriptionResult:
    text: str
    segments: list[Segment]
    language: str
    duration_seconds: float
    wpm: float
    audio_path: str   # путь к WAV файлу — переиспользуется другими модулями


# ─── Step 1: Download audio ───────────────────────────────────────────────────

def download_audio(url: str, output_dir: str = AUDIO_DIR) -> str:
    """
    Скачивает аудиодорожку с YouTube через yt-dlp.
    Возвращает путь к скачанному файлу (обычно .webm или .m4a).
    """
    os.makedirs(output_dir, exist_ok=True)
    output_template = os.path.join(output_dir, "%(id)s.%(ext)s")

    cmd = [
        "yt-dlp",
        "--no-playlist",
        "--extract-audio",
        "--audio-format", "best",       # берём лучшее качество, потом конвертируем ffmpeg
        "--audio-quality", "0",
        "--output", output_template,
        "--print", "after_move:filepath",  # выводит путь к файлу после скачивания
        url,
    ]

    logger.info(f"Скачиваем аудио: {url}")
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)

    # последняя строка вывода — путь к файлу
    downloaded_path = result.stdout.strip().splitlines()[-1]
    logger.info(f"Скачано: {downloaded_path}")
    return downloaded_path


# ─── Step 2: Convert to mono WAV 16kHz ───────────────────────────────────────

def convert_to_wav(input_path: str, output_dir: str = AUDIO_DIR) -> str:
    """
    Конвертирует аудио в моно WAV 16kHz через ffmpeg.
    Требование faster-whisper, DNSMOS, silero-VAD.
    Возвращает путь к WAV файлу.
    """
    input_path = Path(input_path)
    wav_path = Path(output_dir) / (input_path.stem + "_16k.wav")

    cmd = [
        "ffmpeg",
        "-y",                        # перезаписать если существует
        "-i", str(input_path),
        "-ac", "1",                  # моно
        "-ar", "16000",              # 16kHz
        "-sample_fmt", "s16",        # 16-bit PCM
        str(wav_path),
    ]

    logger.info(f"Конвертируем в WAV 16kHz: {wav_path}")
    subprocess.run(cmd, capture_output=True, check=True)
    logger.info(f"WAV готов: {wav_path}")
    return str(wav_path)


# ─── Step 3 + 4: Transcribe + WPM ─────────────────────────────────────────────

def transcribe(wav_path: str) -> TranscriptionResult:
    """
    Транскрибирует WAV файл через faster-whisper.
    Возвращает TranscriptionResult с текстом, сегментами, WPM.
    """
    logger.info(f"Загружаем Whisper {WHISPER_MODEL} на {DEVICE} ({COMPUTE_TYPE})")
    model = WhisperModel(
        WHISPER_MODEL,
        device=DEVICE,
        compute_type=COMPUTE_TYPE,
    )

    logger.info("Транскрибируем...")
    segments_raw, info = model.transcribe(
        wav_path,
        language=WHISPER_LANGUAGE,
        condition_on_previous_text=WHISPER_CONDITION_ON_PREV,
        vad_filter=True,             # встроенный VAD для ускорения — пропускает тишину
        vad_parameters=dict(
            min_silence_duration_ms=500,
        ),
    )

    # Материализуем генератор сегментов
    segments = []
    full_text_parts = []
    for seg in segments_raw:
        segments.append(Segment(
            text=seg.text.strip(),
            start=seg.start,
            end=seg.end,
        ))
        full_text_parts.append(seg.text.strip())

    full_text = " ".join(full_text_parts)
    duration = info.duration  # секунды, из faster-whisper

    # WPM: суммируем слова по всем сегментам / длительность в минутах
    # Используем только сегменты с речью, не всю длину файла
    total_words = sum(seg.word_count for seg in segments)
    speech_duration_min = sum(seg.duration for seg in segments) / 60.0
    wpm = round(total_words / speech_duration_min, 1) if speech_duration_min > 0 else 0.0

    logger.info(
        f"Транскрипция готова: {len(segments)} сегментов, "
        f"{total_words} слов, {wpm} WPM, язык={info.language}"
    )

    return TranscriptionResult(
        text=full_text,
        segments=segments,
        language=info.language,
        duration_seconds=duration,
        wpm=wpm,
        audio_path=wav_path,
    )


# ─── Main entry point ─────────────────────────────────────────────────────────

def run(url: str) -> TranscriptionResult:
    """
    Полный цикл: скачать → конвертировать → транскрибировать.
    """
    raw_path = download_audio(url)
    wav_path = convert_to_wav(raw_path)
    return transcribe(wav_path)

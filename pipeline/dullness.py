"""
Модуль флага «Унылость».

Детектирует монотонность и однородность подачи.
Бинарный флаг (да/нет) + числовой score 0..1.

Основание: Shi et al. (2019, SALMM workshop, ACM MM)
DOI: 10.1145/3343031.3350553

Входные данные:
  - TranscriptionResult  → текст, сегменты, WPM
  - AudioQualityResult   → VAD сегменты (переиспользуем)
  - wav_path             → путь к WAV для parselmouth и librosa
"""

import logging
import re
from dataclasses import dataclass
from typing import Optional

import numpy as np
import soundfile as sf
import librosa
import opensmile

from transcription import TranscriptionResult
from audio_quality import AudioQualityResult, VADSegment
from config import (
    DULLNESS_THRESHOLD,
    DULLNESS_WEIGHT_ACOUSTIC,
    DULLNESS_WEIGHT_LINGUISTIC,
    WPM_TOO_SLOW, WPM_TOO_FAST,
    HESITATION_PAUSE_THRESHOLD,
)

logger = logging.getLogger(__name__)


# ─── Data classes ─────────────────────────────────────────────────────────────

@dataclass
class AcousticFeatures:
    f0_mean: Optional[float]          # средняя высота голоса (Hz)
    f0_std: Optional[float]           # вариативность питча (Hz)
    hnr_log: Optional[float]          # log(HNR) — чистота голоса
    rms_std: Optional[float]          # динамика громкости
    spectral_flux_mean: float         # изменчивость спектра
    wpm: float                        # слов в минуту
    hesitation_pause_ratio: float     # доля длинных пауз


@dataclass
class LinguisticFeatures:
    ttr_global: float       # лексическое разнообразие по всей лекции
    ttr_local: float        # лексическое разнообразие в скользящем окне
    filler_ratio: float     # доля слов-паразитов
    question_ratio: float   # доля вопросительных предложений
    engagement_ratio: float # маркеры взаимодействия с аудиторией
    example_ratio: float    # маркеры примеров и аналогий


@dataclass
class DullnessResult:
    # Итоговый флаг и score
    flag: bool          # True = унылость обнаружена
    score: float        # 0..1, чем выше тем унылее

    # Компоненты
    acoustic: AcousticFeatures
    linguistic: LinguisticFeatures

    # Нормированные sub-scores для отладки
    acoustic_score: float
    linguistic_score: float


# ─── Словари для лингвистики ──────────────────────────────────────────────────

FILLERS_RU = {
    'эм', 'ээ', 'эээ', 'мм', 'ммм', 'ну', 'вот', 'значит',
    'как бы', 'то есть', 'короче', 'типа', 'собственно',
    'в общем', 'ладно', 'слушай', 'понимаешь', 'знаешь',
}

ENGAGEMENT_MARKERS_RU = {
    'вы', 'вам', 'ваш', 'вашего', 'вашей', 'вашу',
    'посмотрите', 'обратите', 'заметьте', 'представьте',
    'давайте', 'попробуем', 'рассмотрим', 'подумайте',
    'вспомните', 'скажите', 'ответьте', 'спросите',
}

EXAMPLE_MARKERS_RU = {
    'например', 'допустим', 'скажем', 'предположим',
    'представьте', 'рассмотрим', 'возьмём', 'это как',
    'аналогично', 'похоже', 'в частности', 'частности',
}


# ─── 1. Акустика: F0 через parselmouth ────────────────────────────────────────

def compute_f0(
    wav_path: str,
    speech_segments: list[VADSegment],
) -> tuple[Optional[float], Optional[float]]:
    """
    F0 mean и F0 std через parselmouth (Praat).
    Считаем только по речевым сегментам.

    Возвращает (None, None) если parselmouth не установлен.
    Shi et al.: F0 std — ключевой признак монотонности.
    """
    try:
        import parselmouth
        from parselmouth.praat import call
    except ImportError:
        logger.warning("parselmouth не установлен — F0 пропускаем")
        return None, None

    logger.info("Считаем F0 через parselmouth...")

    snd = parselmouth.Sound(wav_path)
    pitch = snd.to_pitch(time_step=0.01, pitch_floor=75, pitch_ceiling=500)
    f0_values = pitch.selected_array['frequency']

    # Берём только voiced фреймы (f0 > 0) в речевых сегментах
    times = pitch.xs()
    voiced_f0 = []

    for seg in speech_segments:
        mask = (times >= seg.start) & (times <= seg.end) & (f0_values > 0)
        voiced_f0.extend(f0_values[mask].tolist())

    if len(voiced_f0) < 10:
        logger.warning("F0: недостаточно voiced фреймов")
        return None, None

    f0_arr = np.array(voiced_f0)
    f0_mean = round(float(np.mean(f0_arr)), 2)
    f0_std = round(float(np.std(f0_arr)), 2)

    logger.info(f"F0: mean={f0_mean} Hz, std={f0_std} Hz")
    return f0_mean, f0_std


# ─── 2. Акустика: HNR и RMS через openSMILE eGeMAPS ──────────────────────────

def compute_opensmile_features(
    wav_path: str,
) -> tuple[Optional[float], Optional[float]]:
    """
    log(HNR) и RMS_std из openSMILE eGeMAPS.

    log(HNR) — наибольшая корреляция с качеством лекции (Shi et al.)
    RMS_std  — динамика громкости (Loudness variation по Shi et al.)
    """
    logger.info("Считаем openSMILE eGeMAPS...")

    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.Functionals,
    )

    features = smile.process_file(wav_path)

    # HNR — ищем колонку с HNR в названии
    hnr_cols = [c for c in features.columns if 'HNR' in c or 'hnr' in c.lower()]
    rms_cols = [c for c in features.columns if 'loudness' in c.lower() or 'rms' in c.lower()]

    hnr_log = None
    rms_std = None

    if hnr_cols:
        hnr_val = float(features[hnr_cols[0]].iloc[0])
        # Берём log чтобы сжать диапазон — как в Shi et al.
        hnr_log = round(float(np.log(max(hnr_val, 1e-10))), 4)
        logger.info(f"HNR: {hnr_val:.2f} → log(HNR)={hnr_log:.4f}")
    else:
        logger.warning("HNR колонка не найдена в eGeMAPS")

    # RMS std — ищем колонку со стандартным отклонением громкости
    rms_std_cols = [c for c in rms_cols if 'stddev' in c.lower() or 'std' in c.lower()]
    if rms_std_cols:
        rms_std = round(float(features[rms_std_cols[0]].iloc[0]), 6)
        logger.info(f"RMS_std: {rms_std:.6f}")
    elif rms_cols:
        # Фоллбэк — берём любую loudness колонку
        rms_std = round(float(features[rms_cols[0]].iloc[0]), 6)
        logger.info(f"RMS (fallback): {rms_std:.6f}")
    else:
        logger.warning("RMS/Loudness колонка не найдена в eGeMAPS")

    return hnr_log, rms_std


# ─── 3. Акустика: Spectral flux через librosa ─────────────────────────────────

def compute_spectral_flux(
    audio: np.ndarray,
    sr: int,
    speech_segments: list[VADSegment],
) -> float:
    """
    Spectral flux — изменчивость спектра во времени.
    Низкая → монотонный тембр (Shi et al.)
    Считаем только по речевым сегментам.
    """
    logger.info("Считаем spectral flux...")

    if not speech_segments:
        return 0.0

    speech_audio = np.concatenate([
        audio[int(s.start * sr):int(s.end * sr)]
        for s in speech_segments
    ])

    # onset_strength как прокси для spectral flux
    flux = librosa.onset.onset_strength(y=speech_audio, sr=sr)
    flux_mean = round(float(np.mean(flux)), 4)

    logger.info(f"Spectral flux mean: {flux_mean:.4f}")
    return flux_mean


# ─── 4. Акустика: Hesitation pause ratio ──────────────────────────────────────

def compute_hesitation_pause_ratio(
    speech_segments: list[VADSegment],
) -> float:
    """
    Доля длинных пауз (> HESITATION_PAUSE_THRESHOLD секунд) между сегментами.
    Высокий → неуверенная, прерывистая речь.

    Порог 0.5с: Goldman-Eisler (1968), Levelt (1989)
    """
    if len(speech_segments) < 2:
        return 0.0

    sorted_segs = sorted(speech_segments, key=lambda s: s.start)
    pauses = []

    for i in range(1, len(sorted_segs)):
        pause = sorted_segs[i].start - sorted_segs[i - 1].end
        if pause > 0:
            pauses.append(pause)

    if not pauses:
        return 0.0

    long_pauses = sum(1 for p in pauses if p > HESITATION_PAUSE_THRESHOLD)
    ratio = round(long_pauses / len(pauses), 4)

    logger.info(f"Hesitation pause ratio: {ratio:.4f} ({long_pauses}/{len(pauses)} пауз)")
    return ratio


# ─── 5. Лингвистика ───────────────────────────────────────────────────────────

def compute_ttr_global(words: list[str]) -> float:
    """TTR по всей лекции — лексическое разнообразие."""
    if not words:
        return 0.0
    return round(len(set(words)) / len(words), 4)


def compute_ttr_local(words: list[str], window: int = 50) -> float:
    """
    Среднее TTR в скользящем окне.
    Ловит локальные повторы — лектор крутится вокруг одних слов.
    """
    if len(words) < window:
        return compute_ttr_global(words)

    scores = []
    for i in range(len(words) - window):
        chunk = words[i:i + window]
        scores.append(len(set(chunk)) / window)

    return round(float(np.mean(scores)), 4)


def compute_filler_ratio(words: list[str], text: str) -> float:
    """
    Доля слов-паразитов.
    Ищем однословные и многословные маркеры.
    """
    if not words:
        return 0.0

    text_lower = text.lower()
    count = 0

    # Однословные
    for w in words:
        if w.lower() in FILLERS_RU:
            count += 1

    # Многословные (как бы, то есть, в общем)
    multi_fillers = ['как бы', 'то есть', 'в общем', 'ну вот', 'ну да']
    for mf in multi_fillers:
        count += text_lower.count(mf)

    return round(count / len(words), 4)


def compute_question_ratio(text: str) -> float:
    """
    Доля вопросительных предложений.
    Низкий → нет диалога с аудиторией.
    """
    sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
    if not sentences:
        return 0.0

    questions = text.count('?')
    return round(questions / len(sentences), 4)


def compute_engagement_ratio(words: list[str], text: str) -> float:
    """
    Маркеры взаимодействия с аудиторией.
    Низкий → нет обращений к слушателям.
    """
    if not words:
        return 0.0

    text_lower = text.lower()
    count = sum(1 for w in words if w.lower() in ENGAGEMENT_MARKERS_RU)

    # Многословные маркеры
    multi = ['обратите внимание', 'посмотрите на', 'как вы думаете']
    for m in multi:
        count += text_lower.count(m)

    return round(count / len(words), 4)


def compute_example_ratio(words: list[str], text: str) -> float:
    """
    Маркеры примеров и аналогий.
    Низкий → нет примеров, сухое перечисление фактов.
    """
    if not words:
        return 0.0

    text_lower = text.lower()
    count = sum(1 for w in words if w.lower() in EXAMPLE_MARKERS_RU)

    # Многословные маркеры
    multi = ['например если', 'это как', 'в частности']
    for m in multi:
        count += text_lower.count(m)

    return round(count / len(words), 4)


# ─── 6. Нормализация признаков ────────────────────────────────────────────────

def normalize_acoustic(features: AcousticFeatures) -> float:
    """
    Нормализуем каждый акустический признак в [0..1] где
    1 = максимальная унылость, 0 = минимальная.
    Возвращаем взвешенное среднее.
    """
    scores = []
    weights = []

    # F0 std — низкое = монотонно = уныло
    # Нормальный диапазон ~20–60 Hz std для лекционной речи
    if features.f0_std is not None:
        f0_score = 1.0 - min(features.f0_std / 40.0, 1.0)
        scores.append(f0_score)
        weights.append(1.5)  # чуть больший вес — ключевой признак

    # log(HNR) — низкое = сиплый/невыразительный = уныло
    # Нормальный диапазон log(HNR) примерно 1..4
    if features.hnr_log is not None:
        hnr_score = 1.0 - min(max(features.hnr_log, 0) / 3.0, 1.0)
        scores.append(hnr_score)
        weights.append(2.0)  # наибольший вес по Shi et al.

    # RMS std — низкое = нет динамики = уныло
    if features.rms_std is not None:
        rms_score = 1.0 - min(features.rms_std / 0.1, 1.0)
        scores.append(rms_score)
        weights.append(1.0)

    # Spectral flux — низкое = монотонный тембр = уныло
    if features.spectral_flux_mean > 0:
        flux_score = 1.0 - min(features.spectral_flux_mean / 5.0, 1.0)
        scores.append(flux_score)
        weights.append(1.0)

    # WPM — ниже 100 = усыпляет, выше 180 = тоже плохо но по другой причине
    wpm_score = 0.0
    if features.wpm < WPM_TOO_SLOW:
        wpm_score = 1.0 - (features.wpm / WPM_TOO_SLOW)
    elif features.wpm > WPM_TOO_FAST:
        wpm_score = min((features.wpm - WPM_TOO_FAST) / 60.0, 0.5)  # меньший штраф
    scores.append(wpm_score)
    weights.append(1.0)

    # Hesitation pause ratio — высокое = прерывистая речь = уныло
    pause_score = min(features.hesitation_pause_ratio / 0.5, 1.0)
    scores.append(pause_score)
    weights.append(0.5)

    if not scores:
        return 0.5  # нет данных — нейтрально

    weighted = sum(s * w for s, w in zip(scores, weights)) / sum(weights)
    return round(float(weighted), 4)


def normalize_linguistic(features: LinguisticFeatures) -> float:
    """
    Нормализуем лингвистические признаки в [0..1].
    1 = максимальная унылость.
    """
    scores = []
    weights = []

    # TTR global — низкое = бедная лексика = уныло
    # Нормальный диапазон ~0.3–0.7 для лекций
    ttr_g_score = 1.0 - min(features.ttr_global / 0.5, 1.0)
    scores.append(ttr_g_score)
    weights.append(1.0)

    # TTR local — низкое = локальные повторы = уныло
    ttr_l_score = 1.0 - min(features.ttr_local / 0.5, 1.0)
    scores.append(ttr_l_score)
    weights.append(1.5)  # чуть важнее — ловит локальные паттерны

    # Filler ratio — высокое = много паразитов = уныло
    filler_score = min(features.filler_ratio / 0.05, 1.0)
    scores.append(filler_score)
    weights.append(1.0)

    # Question ratio — низкое = нет диалога = уныло
    # Норма ~0.05–0.15 для хорошей лекции
    q_score = 1.0 - min(features.question_ratio / 0.1, 1.0)
    scores.append(q_score)
    weights.append(1.0)

    # Engagement ratio — низкое = нет взаимодействия = уныло
    eng_score = 1.0 - min(features.engagement_ratio / 0.02, 1.0)
    scores.append(eng_score)
    weights.append(1.0)

    # Example ratio — низкое = нет примеров = уныло
    ex_score = 1.0 - min(features.example_ratio / 0.02, 1.0)
    scores.append(ex_score)
    weights.append(0.5)

    weighted = sum(s * w for s, w in zip(scores, weights)) / sum(weights)
    return round(float(weighted), 4)


# ─── Main entry point ─────────────────────────────────────────────────────────

def run(
    transcription: TranscriptionResult,
    audio_quality: AudioQualityResult,
    wav_path: str,
) -> DullnessResult:
    """
    Полный анализ флага «Унылость».

    Принимает:
      transcription  — результат модуля 1 (текст, сегменты, WPM)
      audio_quality  — результат модуля 2 (VAD сегменты)
      wav_path       — путь к WAV файлу (для parselmouth и librosa)
    """
    logger.info("── Флаг Унылость ──")

    speech_segments = audio_quality.speech_segments

    # Загружаем аудио для librosa
    audio, sr = sf.read(wav_path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    # ── Акустические признаки ──────────────────────────────────────────────

    # F0 через parselmouth
    f0_mean, f0_std = compute_f0(wav_path, speech_segments)

    # HNR и RMS через openSMILE
    hnr_log, rms_std = compute_opensmile_features(wav_path)

    # Spectral flux через librosa
    spectral_flux_mean = compute_spectral_flux(audio, sr, speech_segments)

    # WPM — уже посчитан в транскрипции
    wpm = transcription.wpm

    # Hesitation pause ratio — из VAD сегментов
    hesitation_pause_ratio = compute_hesitation_pause_ratio(speech_segments)

    acoustic = AcousticFeatures(
        f0_mean=f0_mean,
        f0_std=f0_std,
        hnr_log=hnr_log,
        rms_std=rms_std,
        spectral_flux_mean=spectral_flux_mean,
        wpm=wpm,
        hesitation_pause_ratio=hesitation_pause_ratio,
    )

    # ── Лингвистические признаки ───────────────────────────────────────────

    text = transcription.text
    words = [w.lower() for w in text.split() if w.strip()]

    ttr_global = compute_ttr_global(words)
    ttr_local = compute_ttr_local(words)
    filler_ratio = compute_filler_ratio(words, text)
    question_ratio = compute_question_ratio(text)
    engagement_ratio = compute_engagement_ratio(words, text)
    example_ratio = compute_example_ratio(words, text)

    linguistic = LinguisticFeatures(
        ttr_global=ttr_global,
        ttr_local=ttr_local,
        filler_ratio=filler_ratio,
        question_ratio=question_ratio,
        engagement_ratio=engagement_ratio,
        example_ratio=example_ratio,
    )

    # ── Агрегация ──────────────────────────────────────────────────────────

    acoustic_score = normalize_acoustic(acoustic)
    linguistic_score = normalize_linguistic(linguistic)

    # Взвешенная сумма: 60% акустика, 40% лингвистика
    total_score = round(
        acoustic_score * DULLNESS_WEIGHT_ACOUSTIC +
        linguistic_score * DULLNESS_WEIGHT_LINGUISTIC,
        4
    )

    flag = total_score >= DULLNESS_THRESHOLD

    logger.info(
        f"Унылость: acoustic={acoustic_score}, linguistic={linguistic_score}, "
        f"total={total_score}, flag={flag}"
    )

    return DullnessResult(
        flag=flag,
        score=total_score,
        acoustic=acoustic,
        linguistic=linguistic,
        acoustic_score=acoustic_score,
        linguistic_score=linguistic_score,
    )

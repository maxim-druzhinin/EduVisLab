"""
Точка входа пайплайна.

Запуск:
  python pipeline.py --url "https://youtube.com/watch?v=..."
  python pipeline.py --url "..." --output result.json

Или из Python:
  from pipeline import run
  result = run("https://youtube.com/watch?v=...")
"""

import os
import sys
import json
import logging
import argparse
from datetime import datetime
from dataclasses import asdict

from config import OUTPUT_DIR
import transcription as transcription_module
import audio_quality as audio_quality_module

# ─── Logging ──────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
    #colab
    stream=sys.stdout,
    force=True,
)
logger = logging.getLogger("pipeline")


# ─── JSON serialization helpers ───────────────────────────────────────────────

def segments_to_dict(segments):
    return [{"text": s.text, "start": s.start, "end": s.end} for s in segments]

def vad_segments_to_dict(segments):
    return [{"start": s.start, "end": s.end} for s in segments]


# ─── Main pipeline ────────────────────────────────────────────────────────────

def run(url: str) -> dict:
    """
    Запускает полный пайплайн для YouTube-ссылки.
    Возвращает словарь с результатами всех модулей.
    """
    started_at = datetime.now().isoformat()
    logger.info(f"═══ Запуск пайплайна: {url} ═══")

    # ── Модуль 1: Транскрипция ─────────────────────────────────────────────
    logger.info("── Модуль 1: Транскрипция ──")
    tr = transcription_module.run(url)

    # ── Модуль 2: Техническое качество ────────────────────────────────────
    logger.info("── Модуль 2: Техническое качество ──")
    aq = audio_quality_module.run(tr.audio_path)

    # ── Сборка результата ──────────────────────────────────────────────────
    result = {
        "meta": {
            "video_url": url,
            "processed_at": started_at,
            "duration_seconds": round(tr.duration_seconds, 1),
        },
        "transcription": {
            "language": tr.language,
            "wpm": tr.wpm,
            "text": tr.text,
            "segments": segments_to_dict(tr.segments),
        },
        "audio_quality": {
            "dnsmos": {
                "ovrl_mos": aq.ovrl_mos,
                "sig_mos":  aq.sig_mos,
                "bak_mos":  aq.bak_mos,
                "quality":  aq.mos_quality,
            },
            "lufs": {
                "value":   aq.lufs,
                "quality": aq.lufs_quality,
            },
            "clipping": {
                "ratio":    aq.clipping_ratio,
                "crest_factor": aq.crest_factor,
                "flattened_peaks_ratio": aq.flattened_peaks_ratio,
                "level":    aq.clipping_level,   # "none" / "mild" / "severe"
            },
            "snr": {
                "db":      aq.snr_db,
                "quality": aq.snr_quality,
            },
            # VAD сегменты для отладки и переиспользования в модуле Унылость
            "_vad_segments": vad_segments_to_dict(aq.speech_segments),
        },
        # Флаг Унылость — следующий этап
        "dullness": None,
    }

    logger.info("═══ Пайплайн завершён ═══")
    return result


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Аудио-пайплайн анализа образовательного видео"
    )
    parser.add_argument(
        "--url", required=True,
        help="Ссылка на YouTube-видео"
    )
    parser.add_argument(
        "--output", default=None,
        help="Путь для сохранения JSON (опционально)"
    )
    parser.add_argument(
        "--pretty", action="store_true", default=True,
        help="Красивый JSON вывод (по умолчанию включён)"
    )
    args = parser.parse_args()

    result = run(args.url)

    # Вывод в stdout
    output_json = json.dumps(result, ensure_ascii=False, indent=2 if args.pretty else None)
    #print(output_json)

    # Сохранение в файл
    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(output_json)
        logger.info(f"Результат сохранён: {args.output}")


if __name__ == "__main__":
    main()

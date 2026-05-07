"""
Точка входа пайплайна.

Запуск:
  python pipeline.py --url "https://youtube.com/watch?v=..."
  python pipeline.py --url "..." --output result.json
  python pipeline.py --url "..." --skip-video   # только аудио-модули

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
import dullness as dullness_module
import video_quality as video_quality_module

# ─── Logging ──────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
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

def run(
    url: str,
    skip_transcription: bool = False,
    skip_audio: bool = False,
    skip_dullness: bool = False,
    skip_video: bool = False,
) -> dict:
    
    started_at = datetime.now().isoformat()

    tr, aq, dl, vq = None, None, None, None

    if not skip_transcription:
        logger.info("── Модуль 1: Транскрипция ──")
        tr = transcription_module.run(url)

    if not skip_audio:
        logger.info("── Модуль 2: Качество звука ──")
        aq = audio_quality_module.run(tr.audio_path)

    if not skip_dullness:
        logger.info("── Модуль 3: Унылость ──")
        dl = dullness_module.run(tr, aq, tr.audio_path)

    if not skip_video:
        logger.info("── Модуль 4: Качество видео ──")
        video_path = video_quality_module.download_video(url)
        vq = video_quality_module.run(video_path)

    # ── Сборка результата ──────────────────────────────────────────────────
    result = {
        "meta": {
            "video_url":       url,
            "processed_at":    started_at,
            "duration_seconds": round(tr.duration_seconds, 1) if tr else None,
        },
        "transcription": {
            "language": tr.language,
            "wpm":      tr.wpm,
            "text":     tr.text,
            "segments": segments_to_dict(tr.segments),
        } if tr else None,
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
                "ratio":                 aq.clipping_ratio,
                "crest_factor":          aq.crest_factor,
                "flattened_peaks_ratio": aq.flattened_peaks_ratio,
                "level":                 aq.clipping_level,
            },
            "snr": {
                "db":      aq.snr_db,
                "quality": aq.snr_quality,
            },
            "_vad_segments": vad_segments_to_dict(aq.speech_segments),
        } if aq else None,
        "dullness": {
            "flag":  dl.flag,
            "score": dl.score,
            "acoustic_score":   dl.acoustic_score,
            "linguistic_score": dl.linguistic_score,
            "components": {
                "f0_mean":                dl.acoustic.f0_mean,
                "f0_std":                 dl.acoustic.f0_std,
                "hnr_log":                dl.acoustic.hnr_log,
                "rms_std":                dl.acoustic.rms_std,
                "spectral_flux_mean":     dl.acoustic.spectral_flux_mean,
                "wpm":                    dl.acoustic.wpm,
                "hesitation_pause_ratio": dl.acoustic.hesitation_pause_ratio,
                "ttr_global":             dl.linguistic.ttr_global,
                "ttr_local":              dl.linguistic.ttr_local,
                "filler_ratio":           dl.linguistic.filler_ratio,
                "question_ratio":         dl.linguistic.question_ratio,
                "engagement_ratio":       dl.linguistic.engagement_ratio,
                "example_ratio":          dl.linguistic.example_ratio,
            },
        } if dl else None,
        "video_quality": _serialize_video_quality(vq),
    }

    logger.info("═══ Пайплайн завершён ═══")
    return result


def _serialize_video_quality(vq) -> dict:
    """Сериализует VideoQualityResult в словарь для JSON."""
    if vq is None:
        return {"skipped": True}

    return {
        "resolution": {
            "width":   vq.width,
            "height":  vq.height,
            "fps":     vq.fps,
            "codec":   vq.codec,
            "bitrate_kbps": vq.bitrate_kbps,
            "level":   vq.resolution_level,
        },
        "dover": {
            "score":   vq.dover_score,
            "quality": vq.dover_quality,
        },
        "brisque": {
            "median": vq.brisque_median,
            "p90":    vq.brisque_p90,
            "level":  vq.brisque_level,
        },
        "blur": {
            "median": vq.blur_median,
            "p10":    vq.blur_p10,
            "level":  vq.blur_level,
        },
        "exposure": {
            "mean_brightness":    vq.mean_brightness,
            "overexposed_ratio":  vq.overexposed_ratio,
            "underexposed_ratio": vq.underexposed_ratio,
            "level":              vq.exposure_level,
        },
        "contrast": {
            "median": vq.contrast_median,
            "level":  vq.contrast_level,
        },
        "flicker": {
            "mean":  vq.flicker_mean,
            "level": vq.flicker_level,
        },
        "stability": {
            "motion_mean": vq.motion_mean,
            "motion_std":  vq.motion_std,
            "level":       vq.stability_level,
        },
        "consistency": {
            "blur_std":            vq.blur_std,
            "brisque_std":         vq.brisque_std,
            "dover_technical_std": vq.dover_technical_std,
            "brightness_std":      vq.brightness_std,
            "level":               vq.consistency_level,
        },
        "speaker": {
            "face_presence_ratio": vq.face_presence_ratio,
            "face_size_median":    vq.face_size_median,
        },
        "board": {
            "detected":         vq.board_detected,
            "glare_ratio":      vq.board_glare_ratio,
            "contrast":         vq.board_contrast,
            "glare_level":      vq.board_glare_level,
            "readability_level": vq.board_readability_level,
        },
    }


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Аудио-видео пайплайн анализа образовательного видео"
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
    parser.add_argument("--skip-transcription", action="store_true", default=False)
    parser.add_argument("--skip-audio",         action="store_true", default=False)
    parser.add_argument("--skip-dullness",      action="store_true", default=False)
    parser.add_argument("--skip-video",         action="store_true", default=False)
    args = parser.parse_args()

    result = run(
        args.url,
        skip_transcription=args.skip_transcription,
        skip_audio=args.skip_audio,
        skip_dullness=args.skip_dullness,
        skip_video=args.skip_video,
    )

    output_json = json.dumps(result, ensure_ascii=False, indent=2 if args.pretty else None)

    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(output_json)
        logger.info(f"Результат сохранён: {args.output}")


if __name__ == "__main__":
    main()

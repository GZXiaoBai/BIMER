#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import time
import subprocess

from bimer.cli import _runtime_analyzer
from bimer.export import (
    export_analysis_csv,
    export_analysis_figure,
    export_analysis_json,
)


def _timed_analyze(analyzer, video, language):
    started = time.perf_counter()
    result = analyzer.analyze(video, language)
    return result, time.perf_counter() - started


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--yunet-model", required=True)
    parser.add_argument("--chinese-video", required=True, type=Path)
    parser.add_argument("--english-no-face-video", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    analyzer = _runtime_analyzer(
        args.checkpoint,
        args.yunet_model,
        "auto",
        calibration_path=args.calibration,
        cache_directory=str(args.output / "cache"),
        model_version="auto",
    )
    chinese, chinese_seconds = _timed_analyze(
        analyzer,
        args.chinese_video,
        "zh",
    )
    english, english_seconds = _timed_analyze(
        analyzer,
        args.english_no_face_video,
        "en",
    )
    detected, segments = analyzer.transcribe(args.chinese_video, "zh")
    edited = list(segments)
    edited[0] = replace(edited[0], text=edited[0].text + "（人工修改）")
    edit_started = time.perf_counter()
    edited_result = analyzer.analyze_segments(
        args.chinese_video,
        detected_language=detected,
        segments=edited,
    )
    edit_seconds = time.perf_counter() - edit_started

    output = args.output / "exports"
    output.mkdir(parents=True, exist_ok=True)
    export_analysis_json(edited_result, output / "analysis.json")
    export_analysis_csv(edited_result, output / "analysis.csv")
    export_analysis_figure(edited_result, output / "analysis.png")
    no_face_disabled = all(
        not segment.modality_available.get("vision", False)
        for segment in english.segments
    )
    error_root = args.output / "error-inputs"
    error_root.mkdir(parents=True, exist_ok=True)
    wrong_format = error_root / "wrong.txt"
    wrong_format.write_text("not a video", encoding="utf-8")
    oversized = error_root / "oversized.mp4"
    with oversized.open("wb") as handle:
        handle.truncate(500 * 1024 * 1024 + 1)
    silent = error_root / "silent.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=black:s=320x240:d=2",
            "-an",
            "-y",
            str(silent),
        ],
        check=True,
    )

    def rejected(path, expected):
        try:
            analyzer.analyze(path)
        except ValueError as error:
            return expected in str(error)
        return False

    checks = {
        "chinese_under_120_seconds": chinese_seconds <= 120,
        "english_under_120_seconds": english_seconds <= 120,
        "edited_text_under_15_seconds": edit_seconds <= 15,
        "english_no_face_disables_vision": no_face_disabled,
        "json_export": (output / "analysis.json").is_file(),
        "csv_export": (output / "analysis.csv").is_file(),
        "png_export": (output / "analysis.png").is_file(),
        "wrong_format_error": rejected(wrong_format, "MP4 or MOV"),
        "oversized_file_error": rejected(oversized, "500 MB"),
        "silent_video_error": rejected(silent, "audio stream"),
    }
    payload = {
        "checks": checks,
        "passed": all(checks.values()),
        "seconds": {
            "chinese_first": chinese_seconds,
            "english_first": english_seconds,
            "edited_text": edit_seconds,
        },
        "runtime_profiles": {
            "chinese": chinese.runtime_profile,
            "english": english.runtime_profile,
            "edited": edited_result.runtime_profile,
        },
    }
    args.output.mkdir(parents=True, exist_ok=True)
    report = args.output / "m2-acceptance.json"
    report.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(report)
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

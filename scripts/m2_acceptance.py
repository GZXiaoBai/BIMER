#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import time
from dataclasses import replace
from pathlib import Path

from bimer.export import (
    export_analysis_csv,
    export_analysis_figure,
    export_analysis_json,
)
from bimer.inference import TranscriptSegment
from bimer.runtime import build_runtime
from bimer.schema import AnalysisResult


def _timed_analyze(analyzer, video, language):
    started = time.perf_counter()
    result = analyzer.analyze(video, language)
    return result, time.perf_counter() - started


def transcript_segments_from_result(result: AnalysisResult) -> list[TranscriptSegment]:
    return [
        TranscriptSegment(
            start_seconds=segment.start_seconds,
            end_seconds=segment.end_seconds,
            text=segment.text,
        )
        for segment in result.segments
    ]


def chinese_character_ratio(result: AnalysisResult) -> float:
    text = "".join(segment.text for segment in result.segments)
    content = [character for character in text if character.isalnum()]
    if not content:
        return 0.0
    chinese = sum("\u3400" <= character <= "\u9fff" for character in content)
    return chinese / len(content)


def all_segments_have_vision(result: AnalysisResult) -> bool:
    return bool(result.segments) and all(
        segment.modality_available.get("vision", False) for segment in result.segments
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--deployment", required=True, type=Path)
    parser.add_argument("--artifact-root", default=Path("."), type=Path)
    parser.add_argument("--chinese-video", type=Path)
    parser.add_argument("--english-no-face-video", required=True, type=Path)
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Record English-only evidence while leaving final acceptance incomplete.",
    )
    parser.add_argument(
        "--preserve-runtime-cache",
        action="store_true",
        help="Do not clear cached features before the first analysis.",
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    analyzer = build_runtime(
        args.deployment,
        artifact_root=args.artifact_root,
        device_name="auto",
        offline=True,
    )
    runtime_cache = getattr(analyzer.feature_pipeline, "cache", None)
    cleared_cache_entries = 0
    if runtime_cache is not None and not args.preserve_runtime_cache:
        cleared_cache_entries = runtime_cache.clear()
    chinese = None
    chinese_seconds = None
    if args.chinese_video is not None:
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
    edit_video = args.chinese_video or args.english_no_face_video
    edit_source = chinese if chinese is not None else english
    detected = edit_source.language
    edited = transcript_segments_from_result(edit_source)
    edited[0] = replace(edited[0], text=edited[0].text + "（人工修改）")
    edit_started = time.perf_counter()
    edited_result = analyzer.analyze_segments(
        edit_video,
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
        not segment.modality_available.get("vision", False) for segment in english.segments
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
    if chinese_seconds is not None:
        checks["chinese_under_120_seconds"] = chinese_seconds <= 120
        checks["chinese_content_is_chinese"] = chinese_character_ratio(chinese) >= 0.5
        checks["chinese_face_enables_vision"] = all_segments_have_vision(chinese)
    completed_checks_passed = all(checks.values())
    complete = chinese is not None
    payload = {
        "checks": checks,
        "completed_checks_passed": completed_checks_passed,
        "complete": complete,
        "passed": completed_checks_passed and complete,
        "missing_requirements": (
            [] if complete else ["authorized 30-60 second Chinese face video"]
        ),
        "runtime_cache_entries_cleared": cleared_cache_entries,
        "seconds": {
            "chinese_first": chinese_seconds,
            "english_first": english_seconds,
            "edited_text": edit_seconds,
        },
        "runtime_profiles": {
            "chinese": chinese.runtime_profile if chinese is not None else None,
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
    if payload["passed"] or (args.allow_partial and completed_checks_passed):
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

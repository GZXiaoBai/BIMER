#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from bimer.external_annotation_pack import (
    AnnotationSegment,
    build_annotation_rows,
    write_annotation_handoff,
)
from bimer.inference import FasterWhisperTranscriber, normalize_transcript_segments


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Pre-transcribe a locked external-video plan for independent labeling."
    )
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--model", default="small")
    args = parser.parse_args()

    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    transcriber = FasterWhisperTranscriber(args.model, device="cpu")
    by_video: dict[str, list[AnnotationSegment]] = {}
    for video in plan["videos"]:
        language = video["language"]
        _, raw_segments = transcriber.transcribe(Path(video["path"]), language)
        normalized = normalize_transcript_segments(raw_segments)
        by_video[video["video_id"]] = [
            AnnotationSegment(
                start_seconds=segment.start_seconds,
                end_seconds=segment.end_seconds,
                text=segment.text,
                asr_confidence=segment.asr_confidence,
            )
            for segment in normalized
        ]
        print(f"{video['video_id']}: {len(normalized)} segments")

    rows = build_annotation_rows(by_video)
    outputs = write_annotation_handoff(rows, output_dir=args.output_dir)
    print(f"segments: {len(rows)}")
    for path in outputs.values():
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

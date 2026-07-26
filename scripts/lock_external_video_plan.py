#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path

from bimer.external_evaluation import (
    ExternalVideo,
    lock_external_video_plan,
    validate_external_video_plan,
)


def _duration(path: Path) -> float:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return float(result.stdout.strip())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan-csv", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    with args.plan_csv.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    paths = [Path(row["path"]) for row in rows]
    supplied_durations = [row.get("duration_seconds", "").strip() for row in rows]
    measured = [_duration(path) for path in paths]
    for supplied, actual in zip(supplied_durations, measured, strict=True):
        if supplied and abs(float(supplied) - actual) > 1.0:
            raise SystemExit("supplied and measured video durations differ by over one second")
    output = lock_external_video_plan(
        paths,
        languages=[row["language"] for row in rows],
        conditions=[row["condition"] for row in rows],
        durations=measured,
        output_path=args.output,
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    for video, row in zip(payload["videos"], rows, strict=True):
        video["video_id"] = row["video_id"]
    payload["validation"] = validate_external_video_plan(
        [ExternalVideo(**video) for video in payload["videos"]]
    )
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

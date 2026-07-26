from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

from .inference import FasterWhisperTranscriber, RequestedLanguage


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Isolated BIMER Whisper worker")
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--language", choices=("auto", "zh", "en"), default="auto")
    return parser


def run_worker(
    *,
    model: str,
    device: str,
    video: Path,
    language: RequestedLanguage,
) -> dict[str, object]:
    transcriber = FasterWhisperTranscriber(model, device=device)
    detected, segments = transcriber.transcribe(video, language)
    return {
        "language": detected,
        "segments": [asdict(segment) for segment in segments],
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        payload = run_worker(
            model=args.model,
            device=args.device,
            video=args.video,
            language=args.language,
        )
    except Exception as exc:
        print(f"ASR worker failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(payload, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

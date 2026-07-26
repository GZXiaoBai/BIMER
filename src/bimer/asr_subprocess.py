from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Literal

from .inference import RequestedLanguage, TranscriptSegment


class ASRWorkerError(RuntimeError):
    """Raised when the isolated Whisper worker cannot return a valid result."""


class SubprocessWhisperTranscriber:
    """Run Whisper outside the OpenCV process to isolate FFmpeg libraries."""

    def __init__(
        self,
        model_size: str = "small",
        *,
        device: str = "cpu",
        python_executable: str | None = None,
        timeout_seconds: float = 600.0,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self.model_size = model_size
        self.device = device
        self.python_executable = python_executable or sys.executable
        self.timeout_seconds = timeout_seconds

    def transcribe(
        self,
        video_path: Path,
        language: RequestedLanguage,
    ) -> tuple[Literal["zh", "en"], list[TranscriptSegment]]:
        command = [
            self.python_executable,
            "-m",
            "bimer.asr_worker",
            "--model",
            self.model_size,
            "--device",
            self.device,
            "--video",
            str(video_path),
            "--language",
            language,
        ]
        try:
            result = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            raise ASRWorkerError(
                f"Whisper worker timed out after {self.timeout_seconds:g} seconds"
            ) from exc
        if result.returncode != 0:
            detail = result.stderr.strip() or f"exit status {result.returncode}"
            raise ASRWorkerError(f"Whisper worker failed: {detail}")
        try:
            payload = json.loads(result.stdout)
            return _parse_worker_payload(payload)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ASRWorkerError("Whisper worker returned an invalid JSON payload") from exc


def _parse_worker_payload(
    payload: object,
) -> tuple[Literal["zh", "en"], list[TranscriptSegment]]:
    if not isinstance(payload, dict):
        raise TypeError("payload must be an object")
    language = payload["language"]
    if language not in {"zh", "en"}:
        raise ValueError("language must be zh or en")
    raw_segments = payload["segments"]
    if not isinstance(raw_segments, list):
        raise TypeError("segments must be a list")
    segments: list[TranscriptSegment] = []
    for raw in raw_segments:
        if not isinstance(raw, dict):
            raise TypeError("segment must be an object")
        start = float(raw["start_seconds"])
        end = float(raw["end_seconds"])
        text = str(raw["text"]).strip()
        raw_confidence = raw.get("asr_confidence")
        confidence = None if raw_confidence is None else float(raw_confidence)
        if (
            not math.isfinite(start)
            or not math.isfinite(end)
            or start < 0
            or end <= start
            or not text
        ):
            raise ValueError("segment timing and text are invalid")
        if confidence is not None and (
            not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0
        ):
            raise ValueError("segment confidence is invalid")
        segments.append(
            TranscriptSegment(
                start_seconds=start,
                end_seconds=end,
                text=text,
                asr_confidence=confidence,
            )
        )
    return language, segments

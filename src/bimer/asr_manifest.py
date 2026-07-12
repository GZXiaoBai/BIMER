from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Protocol, Sequence

from .inference import TranscriptSegment
from .schema import UtteranceRecord


class ASRTranscriber(Protocol):
    def transcribe(
        self, video_path: Path, language: str
    ) -> tuple[str, Sequence[TranscriptSegment]]: ...


def replace_text_with_asr(
    records: Sequence[UtteranceRecord], transcriber: ASRTranscriber
) -> list[UtteranceRecord]:
    output: list[UtteranceRecord] = []
    for record in records:
        if record.video_path is None:
            raise ValueError(f"record {record.sample_id} has no video for ASR")
        _, segments = transcriber.transcribe(Path(record.video_path), record.language)
        text = " ".join(segment.text.strip() for segment in segments if segment.text.strip()).strip()
        if not text:
            raise ValueError(f"ASR returned no text for {record.sample_id}")
        output.append(replace(record, text=text))
    return output

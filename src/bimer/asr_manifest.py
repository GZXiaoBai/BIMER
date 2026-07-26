from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Protocol, Sequence

from .inference import TranscriptSegment
from .manifest import read_manifest, write_manifest
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
        text = " ".join(
            segment.text.strip() for segment in segments if segment.text.strip()
        ).strip()
        if not text:
            raise ValueError(f"ASR returned no text for {record.sample_id}")
        confidence_rows = [
            float(segment.asr_confidence)
            for segment in segments
            if segment.text.strip() and segment.asr_confidence is not None
        ]
        confidence = sum(confidence_rows) / len(confidence_rows) if confidence_rows else 0.0
        output.append(
            replace(
                record,
                text=text,
                text_source="whisper",
                asr_confidence=confidence,
            )
        )
    return output


def write_asr_manifest_incrementally(
    records: Sequence[UtteranceRecord],
    transcriber: ASRTranscriber,
    output_path: Path | str,
    *,
    keep_original_on_error: bool = False,
    error_path: Path | str | None = None,
) -> list[UtteranceRecord]:
    source = list(records)
    path = Path(output_path)
    completed = read_manifest(path) if path.is_file() else []
    if len(completed) > len(source):
        raise ValueError("existing ASR manifest is longer than the source")
    for index, existing in enumerate(completed):
        if (
            replace(
                existing,
                text=source[index].text,
                text_source=source[index].text_source,
                asr_confidence=source[index].asr_confidence,
            )
            != source[index]
        ):
            raise ValueError(f"existing ASR row {index + 1} does not match the source")
    for record in source[len(completed) :]:
        try:
            replaced = replace_text_with_asr([record], transcriber)[0]
        except Exception as exc:
            if not keep_original_on_error:
                raise
            replaced = record
            if error_path is not None:
                errors = Path(error_path)
                errors.parent.mkdir(parents=True, exist_ok=True)
                with errors.open("a", encoding="utf-8") as handle:
                    handle.write(
                        json.dumps(
                            {
                                "dataset": record.dataset,
                                "error": str(exc),
                                "error_type": type(exc).__name__,
                                "sample_id": record.sample_id,
                                "split": str(record.split),
                            },
                            ensure_ascii=False,
                            sort_keys=True,
                        )
                        + "\n"
                    )
        write_manifest([replaced], path, append=True)
        completed.append(replaced)
    return completed

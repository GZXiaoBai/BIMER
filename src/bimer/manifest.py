from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from .schema import UtteranceRecord


def write_manifest(
    records: Iterable[UtteranceRecord], output_path: Path | str
) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            payload = {
                "dataset": record.dataset,
                "split": record.split,
                "dialogue_id": record.dialogue_id,
                "utterance_id": record.utterance_id,
                "text": record.text,
                "emotion": record.emotion,
                "language": record.language,
                "start_seconds": record.start_seconds,
                "end_seconds": record.end_seconds,
                "speaker_id": record.speaker_id,
                "video_path": str(record.video_path) if record.video_path else None,
                "audio_path": str(record.audio_path) if record.audio_path else None,
            }
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    return path


def read_manifest(path: Path | str) -> list[UtteranceRecord]:
    records: list[UtteranceRecord] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
                records.append(UtteranceRecord(**payload))
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(f"Invalid manifest row {line_number}: {exc}") from exc
    return records


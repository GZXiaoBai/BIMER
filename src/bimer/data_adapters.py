from __future__ import annotations

import json
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from .schema import UtteranceRecord

EXPECTED_SPLIT_COUNTS: dict[str, dict[str, int]] = {
    "meld": {"train": 9989, "dev": 1109, "test": 2610},
    "emotiontalk": {"train": 15413, "validation": 1908, "test": 1929},
}


def parse_timestamp(value: str | float | int) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    normalized = value.strip().replace(",", ".")
    parts = normalized.split(":")
    if len(parts) != 3:
        return float(normalized)
    hours, minutes, seconds = parts
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def probe_duration(path: Path) -> float:
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


def load_meld_csv(
    csv_path: Path | str,
    *,
    media_root: Path | str,
    split: str,
) -> list[UtteranceRecord]:
    table = pd.read_csv(csv_path)
    required = {
        "Utterance",
        "Speaker",
        "Emotion",
        "Dialogue_ID",
        "Utterance_ID",
        "StartTime",
        "EndTime",
    }
    missing = required - set(table.columns)
    if missing:
        raise ValueError(f"MELD CSV missing columns: {sorted(missing)}")
    root = Path(media_root)
    records: list[UtteranceRecord] = []
    for row in table.to_dict(orient="records"):
        dialogue_id = str(row["Dialogue_ID"])
        utterance_id = int(row["Utterance_ID"])
        records.append(
            UtteranceRecord(
                dataset="meld",
                split=split,
                dialogue_id=dialogue_id,
                utterance_id=utterance_id,
                text=str(row["Utterance"]),
                emotion=str(row["Emotion"]),
                language="en",
                speaker_id=str(row["Speaker"]),
                start_seconds=parse_timestamp(row["StartTime"]),
                end_seconds=parse_timestamp(row["EndTime"]),
                video_path=root / f"dia{dialogue_id}_utt{utterance_id}.mp4",
            )
        )
    return records


def _read_json_records(path: Path) -> list[dict[str, Any]]:
    content = path.read_text(encoding="utf-8").strip()
    if not content:
        return []
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in content.splitlines() if line.strip()]
    payload = json.loads(content)
    if isinstance(payload, list):
        return payload
    for key in ("records", "items", "data"):
        if isinstance(payload.get(key), list):
            return payload[key]
    raise ValueError("EmotionTalk manifest must contain a list of records")


def _emotion_from_payload(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict) and value:
        numeric = {key: score for key, score in value.items() if isinstance(score, (int, float))}
        if numeric:
            return max(numeric, key=numeric.get)  # type: ignore[arg-type]
    if isinstance(value, list) and value:
        strings = [str(item) for item in value if isinstance(item, str)]
        if strings:
            return Counter(strings).most_common(1)[0][0]
    raise ValueError(f"Cannot derive emotion from {value!r}")


def _record_times(item: dict[str, Any], media_path: Path) -> tuple[float, float]:
    start = parse_timestamp(item.get("start_seconds", item.get("start", 0.0)))
    if "end_seconds" in item or "end" in item:
        end = parse_timestamp(item.get("end_seconds", item.get("end")))
    elif isinstance(item.get("paragraphs"), dict):
        paragraph = item["paragraphs"]
        start = parse_timestamp(paragraph.get("start", start))
        end = parse_timestamp(paragraph["end"])
    elif "duration_seconds" in item:
        end = start + float(item["duration_seconds"])
    elif media_path.exists():
        end = start + probe_duration(media_path)
    else:
        raise ValueError(f"Missing duration for EmotionTalk media {media_path}")
    return start, end


def load_emotiontalk_manifest(
    manifest_path: Path | str,
    *,
    media_root: Path | str,
    split: str,
) -> list[UtteranceRecord]:
    root = Path(media_root)
    records: list[UtteranceRecord] = []
    for ordinal, item in enumerate(_read_json_records(Path(manifest_path))):
        file_name = str(item["file_name"])
        media_path = root / file_name
        dialogue_id = str(item.get("dialogue_id") or Path(file_name).stem.rsplit("_", 1)[0])
        utterance_id = int(item.get("utterance_id", ordinal))
        start, end = _record_times(item, media_path)
        records.append(
            UtteranceRecord(
                dataset="emotiontalk",
                split=split,
                dialogue_id=dialogue_id,
                utterance_id=utterance_id,
                text=str(item.get("content", "")),
                emotion=_emotion_from_payload(item["emotion_result"]),
                language="zh",
                speaker_id=str(item["speaker_id"]) if item.get("speaker_id") else None,
                start_seconds=start,
                end_seconds=end,
                video_path=media_path,
            )
        )
    return records


def check_official_split_counts(dataset: str, actual: dict[str, int]) -> None:
    if dataset not in EXPECTED_SPLIT_COUNTS:
        raise ValueError(f"Unknown dataset {dataset!r}")
    for split, expected in EXPECTED_SPLIT_COUNTS[dataset].items():
        found = actual.get(split, 0)
        if found != expected:
            raise ValueError(
                f"{dataset} {split}: expected {expected}, found {found}"
            )


def count_records(records: Iterable[UtteranceRecord]) -> dict[str, int]:
    return dict(Counter(str(record.split) for record in records))


from __future__ import annotations

import csv
import json
from pathlib import Path

from .labels import EMOTION_LABELS
from .schema import AnalysisResult


def export_analysis_json(result: AnalysisResult, output_path: Path | str) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(result.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path


def export_analysis_csv(result: AnalysisResult, output_path: Path | str) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "start_seconds",
        "end_seconds",
        "text",
        "emotion",
        *[f"probability_{label}" for label in EMOTION_LABELS],
        "gate_text",
        "gate_audio",
        "gate_vision",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for segment in result.segments:
            row: dict[str, object] = {
                "start_seconds": segment.start_seconds,
                "end_seconds": segment.end_seconds,
                "text": segment.text,
                "emotion": segment.emotion,
                "gate_text": segment.modality_gates.get("text", 0.0),
                "gate_audio": segment.modality_gates.get("audio", 0.0),
                "gate_vision": segment.modality_gates.get("vision", 0.0),
            }
            for label in EMOTION_LABELS:
                row[f"probability_{label}"] = segment.probabilities.get(label, 0.0)
            writer.writerow(row)
    return path

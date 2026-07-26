from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .labels import EMOTION_LABELS
from .quality import MODALITY_QUALITY_NAMES
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
        "confidence_status",
        "calibration_temperature",
        *[f"probability_{label}" for label in EMOTION_LABELS],
        *[f"raw_probability_{label}" for label in EMOTION_LABELS],
        "gate_text",
        "gate_audio",
        "gate_vision",
        *[f"available_{name}" for name in ("text", "audio", "vision")],
        *[
            f"quality_{name}_{field}"
            for name in ("text", "audio", "vision")
            for field in MODALITY_QUALITY_NAMES[name]
        ],
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
                "confidence_status": segment.confidence_status,
                "calibration_temperature": segment.calibration_temperature,
                "gate_text": segment.modality_gates.get("text", 0.0),
                "gate_audio": segment.modality_gates.get("audio", 0.0),
                "gate_vision": segment.modality_gates.get("vision", 0.0),
            }
            for name in ("text", "audio", "vision"):
                row[f"available_{name}"] = segment.modality_available.get(name, False)
                for field in MODALITY_QUALITY_NAMES[name]:
                    row[f"quality_{name}_{field}"] = segment.modality_quality.get(
                        name, {}
                    ).get(field, 0.0)
            for label in EMOTION_LABELS:
                row[f"probability_{label}"] = segment.probabilities.get(label, 0.0)
                row[f"raw_probability_{label}"] = segment.raw_probabilities.get(label, 0.0)
            writer.writerow(row)
    return path


def export_analysis_figure(result: AnalysisResult, output_path: Path | str) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(3, 1, figsize=(12, 9), constrained_layout=True)
    labels = list(EMOTION_LABELS)
    axes[0].bar(labels, [result.global_distribution[label] for label in labels])
    axes[0].set_ylim(0.0, 1.0)
    axes[0].set_title("Global emotion probability distribution")
    for segment in result.segments:
        axes[1].axvspan(
            segment.start_seconds,
            segment.end_seconds,
            alpha=0.35,
            label=str(segment.emotion),
        )
    axes[1].set_title("Emotion timeline")
    axes[1].set_xlabel("Time (seconds)")
    modality_names = ("text", "audio", "vision")
    quality_means = []
    gate_means = []
    for name in modality_names:
        quality_values = [
            value
            for segment in result.segments
            for value in segment.modality_quality.get(name, {}).values()
        ]
        quality_means.append(float(np.mean(quality_values)) if quality_values else 0.0)
        gate_means.append(
            float(np.mean([segment.modality_gates.get(name, 0.0) for segment in result.segments]))
            if result.segments
            else 0.0
        )
    positions = np.arange(3)
    axes[2].bar(positions - 0.18, gate_means, width=0.36, label="gate")
    axes[2].bar(positions + 0.18, quality_means, width=0.36, label="quality")
    axes[2].set_xticks(positions, modality_names)
    axes[2].set_ylim(0.0, 1.0)
    axes[2].set_title("Modality gate and quality")
    axes[2].legend()
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return path

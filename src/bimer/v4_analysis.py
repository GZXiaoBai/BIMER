from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .metrics import classification_metrics

CONTEXT_STRATA = (
    ("1-8", 1, 8),
    ("9-16", 9, 16),
    ("17-32", 17, 32),
)


def ensemble_predictions(paths: Sequence[Path | str]) -> dict[str, np.ndarray]:
    if len(paths) < 2:
        raise ValueError("at least two seed prediction files are required")
    loaded: list[dict[str, np.ndarray]] = []
    for path in paths:
        with np.load(Path(path), allow_pickle=False) as payload:
            required = {"sample_ids", "context_ids", "truth", "probabilities"}
            if not required.issubset(payload.files):
                raise ValueError(f"prediction file {path} is missing required arrays")
            loaded.append({name: payload[name].copy() for name in payload.files})
    reference = loaded[0]
    for candidate in loaded[1:]:
        if (
            not np.array_equal(candidate["sample_ids"], reference["sample_ids"])
            or not np.array_equal(candidate["context_ids"], reference["context_ids"])
            or not np.array_equal(candidate["truth"], reference["truth"])
        ):
            raise ValueError("seed predictions do not align by sample, context, and truth")
        if candidate["probabilities"].shape != reference["probabilities"].shape:
            raise ValueError("seed probability matrices do not align")
    probabilities = np.mean(
        np.stack([payload["probabilities"] for payload in loaded], axis=0),
        axis=0,
        dtype=np.float64,
    )
    probability_sums = probabilities.sum(axis=1, keepdims=True)
    if not np.isfinite(probabilities).all() or np.any(probability_sums <= 0):
        raise ValueError("ensemble probabilities must be finite and non-zero")
    probabilities = probabilities / probability_sums
    return {
        "sample_ids": reference["sample_ids"],
        "context_ids": reference["context_ids"],
        "truth": reference["truth"],
        "probabilities": probabilities.astype(np.float32),
        "prediction": probabilities.argmax(axis=1).astype(np.int64),
    }


def should_enable_ensemble(
    *,
    single: Mapping[str, float],
    ensemble: Mapping[str, float],
) -> bool:
    return (
        float(ensemble["weighted_f1"]) - float(single["weighted_f1"]) >= 0.003 - 1e-12
        and float(ensemble["macro_f1"]) >= float(single["macro_f1"]) - 1e-12
    )


def context_stratified_metrics(
    truth: np.ndarray,
    prediction: np.ndarray,
    context_lengths: np.ndarray,
    *,
    label_names: Sequence[str],
) -> dict[str, dict[str, object]]:
    lengths = {len(truth), len(prediction), len(context_lengths)}
    if len(lengths) != 1 or not len(truth):
        raise ValueError("truth, prediction, and context lengths must align and be non-empty")
    report: dict[str, dict[str, object]] = {}
    for name, lower, upper in CONTEXT_STRATA:
        selected = (context_lengths >= lower) & (context_lengths <= upper)
        if not selected.any():
            report[name] = {"samples": 0}
            continue
        report[name] = {
            "samples": int(selected.sum()),
            **classification_metrics(
                truth[selected],
                prediction[selected],
                label_names=label_names,
            ),
        }
    return report


def _safe_group(values: np.ndarray, selected: np.ndarray) -> dict[str, float | int]:
    if not selected.any():
        return {"samples": 0, "mean_gate": 0.0}
    return {
        "samples": int(selected.sum()),
        "mean_gate": float(values[selected].mean()),
    }


def analyze_context_gates(
    context_gates: np.ndarray,
    context_lengths: np.ndarray,
    local_prediction: np.ndarray,
    fixed_context_prediction: np.ndarray,
) -> dict[str, object]:
    gates = np.asarray(context_gates, dtype=np.float64).reshape(-1)
    lengths = {
        len(gates),
        len(context_lengths),
        len(local_prediction),
        len(fixed_context_prediction),
    }
    if len(lengths) != 1 or not len(gates):
        raise ValueError("context evidence arrays must align and be non-empty")
    if not np.isfinite(gates).all() or np.any((gates < 0) | (gates > 1)):
        raise ValueError("context gates must be finite and within [0, 1]")
    conflicts = local_prediction != fixed_context_prediction
    by_length = {}
    for name, lower, upper in CONTEXT_STRATA:
        selected = (context_lengths >= lower) & (context_lengths <= upper)
        by_length[name] = float(gates[selected].mean()) if selected.any() else None
    return {
        "mean_gate": float(gates.mean()),
        "mean_gate_by_length": by_length,
        "agreement": _safe_group(gates, ~conflicts),
        "conflict": _safe_group(gates, conflicts),
    }


def prototype_geometry(
    representations: np.ndarray,
    truth: np.ndarray,
    *,
    num_classes: int,
) -> dict[str, object]:
    values = np.asarray(representations, dtype=np.float64)
    labels = np.asarray(truth, dtype=np.int64)
    if values.ndim != 2 or labels.shape != (values.shape[0],) or not len(labels):
        raise ValueError("representations and truth must align and be non-empty")
    if not np.isfinite(values).all():
        raise ValueError("representations must be finite")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    normalized = values / np.maximum(norms, np.finfo(np.float64).eps)
    centroids: list[np.ndarray] = []
    within: list[float] = []
    class_details: dict[str, dict[str, float | int]] = {}
    for label in range(num_classes):
        selected = labels == label
        if not selected.any():
            continue
        centroid = normalized[selected].mean(axis=0)
        centroid /= max(np.linalg.norm(centroid), np.finfo(np.float64).eps)
        distances = 1.0 - normalized[selected] @ centroid
        class_distance = float(distances.mean())
        centroids.append(centroid)
        within.extend(distances.tolist())
        class_details[str(label)] = {
            "samples": int(selected.sum()),
            "within_class_distance": class_distance,
        }
    if len(centroids) < 2:
        raise ValueError("prototype geometry requires at least two observed classes")
    between = [
        float(1.0 - centroids[first] @ centroids[second])
        for first in range(len(centroids))
        for second in range(first + 1, len(centroids))
    ]
    within_mean = float(np.mean(within))
    between_mean = float(np.mean(between))
    return {
        "within_class_distance": within_mean,
        "between_class_distance": between_mean,
        "separation": between_mean - within_mean,
        "classes": class_details,
    }

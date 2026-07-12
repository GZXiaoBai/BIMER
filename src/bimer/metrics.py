from __future__ import annotations

from typing import Sequence

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score


def classification_metrics(
    truth: np.ndarray,
    prediction: np.ndarray,
    *,
    label_names: Sequence[str],
) -> dict[str, object]:
    labels = np.arange(len(label_names))
    per_class = f1_score(
        truth,
        prediction,
        labels=labels,
        average=None,
        zero_division=0,
    )
    return {
        "weighted_f1": float(f1_score(truth, prediction, average="weighted", zero_division=0)),
        "macro_f1": float(
            f1_score(truth, prediction, labels=labels, average="macro", zero_division=0)
        ),
        "accuracy": float(accuracy_score(truth, prediction)),
        "per_class_f1": {
            name: float(score) for name, score in zip(label_names, per_class, strict=True)
        },
        "confusion_matrix": confusion_matrix(truth, prediction, labels=labels).tolist(),
    }


def bootstrap_weighted_f1(
    truth: np.ndarray,
    prediction: np.ndarray,
    *,
    iterations: int = 2000,
    seed: int = 42,
) -> tuple[float, float]:
    if len(truth) != len(prediction) or len(truth) == 0:
        raise ValueError("truth and prediction must have equal non-zero length")
    generator = np.random.default_rng(seed)
    scores = np.empty(iterations, dtype=np.float64)
    for index in range(iterations):
        sample = generator.integers(0, len(truth), size=len(truth))
        scores[index] = f1_score(
            truth[sample], prediction[sample], average="weighted", zero_division=0
        )
    low, high = np.quantile(scores, [0.025, 0.975])
    return float(low), float(high)


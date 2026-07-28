from __future__ import annotations

from collections.abc import Iterator
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


def _cluster_bootstrap_indices(
    cluster_ids: np.ndarray,
    *,
    iterations: int,
    seed: int,
) -> Iterator[np.ndarray]:
    clusters = np.asarray(cluster_ids).astype(str)
    unique = np.unique(clusters)
    if not len(unique):
        raise ValueError("cluster_ids must not be empty")
    members = [np.flatnonzero(clusters == cluster) for cluster in unique]
    generator = np.random.default_rng(seed)
    for _ in range(iterations):
        selected = generator.integers(0, len(unique), size=len(unique))
        yield np.concatenate([members[index] for index in selected])


def cluster_bootstrap_weighted_f1(
    truth: np.ndarray,
    prediction: np.ndarray,
    cluster_ids: np.ndarray,
    *,
    iterations: int = 2000,
    seed: int = 42,
) -> tuple[float, float]:
    if len(truth) != len(prediction) or len(truth) != len(cluster_ids) or len(truth) == 0:
        raise ValueError("truth, prediction, and cluster_ids must have equal non-zero length")
    scores = np.asarray(
        [
            f1_score(
                truth[sample],
                prediction[sample],
                average="weighted",
                zero_division=0,
            )
            for sample in _cluster_bootstrap_indices(cluster_ids, iterations=iterations, seed=seed)
        ],
        dtype=np.float64,
    )
    low, high = np.quantile(scores, [0.025, 0.975])
    return float(low), float(high)


def paired_cluster_bootstrap_weighted_f1_delta(
    truth: np.ndarray,
    baseline_prediction: np.ndarray,
    candidate_prediction: np.ndarray,
    cluster_ids: np.ndarray,
    *,
    iterations: int = 2000,
    seed: int = 42,
) -> tuple[float, float]:
    lengths = {len(truth), len(baseline_prediction), len(candidate_prediction), len(cluster_ids)}
    if len(lengths) != 1 or not len(truth):
        raise ValueError("paired bootstrap arrays must have equal non-zero length")
    deltas = []
    for sample in _cluster_bootstrap_indices(cluster_ids, iterations=iterations, seed=seed):
        baseline = f1_score(
            truth[sample], baseline_prediction[sample], average="weighted", zero_division=0
        )
        candidate = f1_score(
            truth[sample], candidate_prediction[sample], average="weighted", zero_division=0
        )
        deltas.append(candidate - baseline)
    low, high = np.quantile(np.asarray(deltas, dtype=np.float64), [0.025, 0.975])
    return float(low), float(high)

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SEEDS = (42, 123, 2026)
DATASETS = ("meld", "emotiontalk")
COMPARISONS = (
    ("formal", "early_mlp"),
    ("formal", "early_context"),
    ("formal", "lagf_no_gates"),
    ("ablations", "no_language"),
    ("ablations", "no_gates"),
    ("ablations", "no_context"),
    ("ablations", "no_quality"),
    ("ablations", "no_modality_dropout"),
    ("ablations", "no_perturbation_training"),
)
CLASS_COUNT = 7


@dataclass(frozen=True)
class PredictionBundle:
    sample_ids: np.ndarray
    context_ids: np.ndarray
    truth: np.ndarray
    prediction: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260725)
    return parser.parse_args()


def _prediction_path(
    root: Path,
    *,
    scope: str,
    variant: str,
    seed: int,
    dataset: str,
) -> Path:
    matches = sorted((root / scope / variant).glob(f"**/seed-{seed}/predictions/{dataset}.npz"))
    if len(matches) != 1:
        raise ValueError(
            f"missing predictions for {scope}/{variant}/seed-{seed}/{dataset}: found {len(matches)}"
        )
    return matches[0]


def _load_bundle(path: Path) -> PredictionBundle:
    with np.load(path, allow_pickle=False) as payload:
        required = {"sample_ids", "context_ids", "truth", "prediction"}
        missing = required.difference(payload.files)
        if missing:
            raise ValueError(f"missing arrays {sorted(missing)} in {path}")
        bundle = PredictionBundle(
            sample_ids=np.asarray(payload["sample_ids"]).astype(str),
            context_ids=np.asarray(payload["context_ids"]).astype(str),
            truth=np.asarray(payload["truth"], dtype=np.int64),
            prediction=np.asarray(payload["prediction"], dtype=np.int64),
        )
    lengths = {
        len(bundle.sample_ids),
        len(bundle.context_ids),
        len(bundle.truth),
        len(bundle.prediction),
    }
    if len(lengths) != 1:
        raise ValueError(f"array length mismatch in {path}")
    if len(set(bundle.sample_ids.tolist())) != len(bundle.sample_ids):
        raise ValueError(f"duplicate sample ids in {path}")
    for values, name in (
        (bundle.truth, "truth"),
        (bundle.prediction, "prediction"),
    ):
        if np.any(values < 0) or np.any(values >= CLASS_COUNT):
            raise ValueError(f"{name} outside seven-class range in {path}")
    return bundle


def _align_pair(
    full: PredictionBundle,
    comparator: PredictionBundle,
    *,
    label: str,
) -> tuple[PredictionBundle, PredictionBundle]:
    full_ids = full.sample_ids.tolist()
    comparator_ids = comparator.sample_ids.tolist()
    if set(full_ids) != set(comparator_ids):
        raise ValueError(f"sample ids do not match for {label}")
    comparator_index = {sample_id: index for index, sample_id in enumerate(comparator_ids)}
    order = np.asarray(
        [comparator_index[sample_id] for sample_id in full_ids],
        dtype=np.int64,
    )
    aligned = PredictionBundle(
        sample_ids=comparator.sample_ids[order],
        context_ids=comparator.context_ids[order],
        truth=comparator.truth[order],
        prediction=comparator.prediction[order],
    )
    if not np.array_equal(full.truth, aligned.truth):
        raise ValueError(f"truth labels do not match for {label}")
    if not np.array_equal(full.context_ids, aligned.context_ids):
        raise ValueError(f"context ids do not match for {label}")
    return full, aligned


def _cluster_confusions(bundle: PredictionBundle) -> np.ndarray:
    context_ids, inverse = np.unique(bundle.context_ids, return_inverse=True)
    confusions = np.zeros(
        (len(context_ids), CLASS_COUNT, CLASS_COUNT),
        dtype=np.int64,
    )
    np.add.at(
        confusions,
        (inverse, bundle.truth, bundle.prediction),
        1,
    )
    return confusions


def _weighted_f1_from_confusions(confusions: np.ndarray) -> np.ndarray:
    matrices = np.asarray(confusions, dtype=np.float64)
    if matrices.ndim == 2:
        matrices = matrices[None, :, :]
    true_support = matrices.sum(axis=2)
    predicted_support = matrices.sum(axis=1)
    true_positive = np.diagonal(matrices, axis1=1, axis2=2)
    denominator = true_support + predicted_support
    per_class = np.divide(
        2 * true_positive,
        denominator,
        out=np.zeros_like(true_positive),
        where=denominator > 0,
    )
    total = true_support.sum(axis=1)
    return np.divide(
        (per_class * true_support).sum(axis=1),
        total,
        out=np.zeros_like(total),
        where=total > 0,
    )


def _paired_delta_draws(
    full: PredictionBundle,
    comparator: PredictionBundle,
    *,
    iterations: int,
    rng: np.random.Generator,
) -> tuple[float, np.ndarray, int]:
    full_confusions = _cluster_confusions(full)
    comparator_confusions = _cluster_confusions(comparator)
    if full_confusions.shape != comparator_confusions.shape:
        raise ValueError("paired models have different context counts")
    cluster_count = len(full_confusions)
    probabilities = np.full(cluster_count, 1 / cluster_count)
    counts = rng.multinomial(
        cluster_count,
        probabilities,
        size=iterations,
    )
    full_draw_confusions = np.einsum(
        "bc,cij->bij",
        counts,
        full_confusions,
        optimize=True,
    )
    comparator_draw_confusions = np.einsum(
        "bc,cij->bij",
        counts,
        comparator_confusions,
        optimize=True,
    )
    draws = _weighted_f1_from_confusions(full_draw_confusions) - _weighted_f1_from_confusions(
        comparator_draw_confusions
    )
    point = float(
        _weighted_f1_from_confusions(full_confusions.sum(axis=0))[0]
        - _weighted_f1_from_confusions(comparator_confusions.sum(axis=0))[0]
    )
    return point, draws, cluster_count


def _comparison_rows(
    root: Path,
    *,
    scope: str,
    comparator: str,
    iterations: int,
    rng: np.random.Generator,
) -> list[dict[str, object]]:
    dataset_points: dict[str, list[float]] = {dataset: [] for dataset in DATASETS}
    dataset_draws: dict[str, list[np.ndarray]] = {dataset: [] for dataset in DATASETS}
    cluster_counts: dict[str, list[int]] = {dataset: [] for dataset in DATASETS}
    for dataset in DATASETS:
        for seed in SEEDS:
            full_path = _prediction_path(
                root,
                scope="formal",
                variant="quality_lagf",
                seed=seed,
                dataset=dataset,
            )
            comparator_path = _prediction_path(
                root,
                scope=scope,
                variant=comparator,
                seed=seed,
                dataset=dataset,
            )
            full, baseline = _align_pair(
                _load_bundle(full_path),
                _load_bundle(comparator_path),
                label=f"{scope}/{comparator}/seed-{seed}/{dataset}",
            )
            point, draws, cluster_count = _paired_delta_draws(
                full,
                baseline,
                iterations=iterations,
                rng=rng,
            )
            dataset_points[dataset].append(point)
            dataset_draws[dataset].append(draws)
            cluster_counts[dataset].append(cluster_count)

    rows: list[dict[str, object]] = []
    for dataset in DATASETS:
        draws = np.mean(np.stack(dataset_draws[dataset]), axis=0)
        rows.append(
            _build_row(
                scope=scope,
                comparator=comparator,
                dataset=dataset,
                point=float(np.mean(dataset_points[dataset])),
                draws=draws,
                iterations=iterations,
                cluster_counts=cluster_counts[dataset],
            )
        )
    bilingual_draws = np.mean(
        np.stack([draw for dataset in DATASETS for draw in dataset_draws[dataset]]),
        axis=0,
    )
    rows.append(
        _build_row(
            scope=scope,
            comparator=comparator,
            dataset="bilingual_average",
            point=float(
                np.mean([point for dataset in DATASETS for point in dataset_points[dataset]])
            ),
            draws=bilingual_draws,
            iterations=iterations,
            cluster_counts=[count for dataset in DATASETS for count in cluster_counts[dataset]],
        )
    )
    return rows


def _build_row(
    *,
    scope: str,
    comparator: str,
    dataset: str,
    point: float,
    draws: np.ndarray,
    iterations: int,
    cluster_counts: list[int],
) -> dict[str, object]:
    lower, upper = np.quantile(draws, [0.025, 0.975])
    return {
        "comparison_scope": ("formal" if scope == "formal" else "ablation"),
        "full_model": "quality_lagf",
        "comparator": comparator,
        "dataset": dataset,
        "seed_count": len(SEEDS),
        "cluster_bootstrap_iterations": iterations,
        "weighted_f1_delta_full_minus_comparator": round(point, 12),
        "ci95_lower": round(float(lower), 12),
        "ci95_upper": round(float(upper), 12),
        "supports_full": bool(lower > 0),
        "cluster_count_min": min(cluster_counts),
        "cluster_count_max": max(cluster_counts),
    }


def main() -> int:
    args = parse_args()
    if args.iterations < 1:
        raise ValueError("iterations must be positive")
    rng = np.random.default_rng(args.bootstrap_seed)
    rows: list[dict[str, object]] = []
    for scope, comparator in COMPARISONS:
        rows.extend(
            _comparison_rows(
                args.input,
                scope=scope,
                comparator=comparator,
                iterations=args.iterations,
                rng=rng,
            )
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)

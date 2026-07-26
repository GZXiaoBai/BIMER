#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

SEEDS = (42, 123, 2026)
DATASETS = ("meld", "emotiontalk")
VARIANTS = ("v3_loss_only", "v3_ranked")


def _result(root: Path, variant: str, seed: int) -> Path:
    return root / f"{variant}-seed-{seed}.json"


def _prediction(root: Path, variant: str, seed: int, dataset: str) -> Path:
    result = _result(root, variant, seed)
    return result.parent / f"{result.stem}.predictions" / f"{dataset}.npz"


def _load_aligned(baseline_path: Path, candidate_path: Path):
    with np.load(baseline_path, allow_pickle=False) as baseline, np.load(
        candidate_path,
        allow_pickle=False,
    ) as candidate:
        baseline_ids = baseline["sample_ids"].astype(str)
        candidate_ids = candidate["sample_ids"].astype(str)
        index = {sample_id: position for position, sample_id in enumerate(candidate_ids)}
        if set(baseline_ids) != set(candidate_ids):
            raise ValueError("V3 formal prediction sample ids do not align")
        order = np.asarray([index[sample_id] for sample_id in baseline_ids])
        truth = baseline["truth"].astype(np.int64)
        if not np.array_equal(truth, candidate["truth"][order]):
            raise ValueError("V3 formal truth labels do not align")
        return (
            truth,
            baseline["prediction"].astype(np.int64),
            candidate["prediction"][order].astype(np.int64),
            baseline["context_ids"].astype(str),
        )


def _paired_draws(truth, baseline, candidate, clusters, *, iterations, seed):
    unique = np.unique(clusters)
    members = [np.flatnonzero(clusters == cluster) for cluster in unique]
    random = np.random.default_rng(seed)
    draws = []
    for _ in range(iterations):
        sampled = random.integers(0, len(unique), len(unique))
        indices = np.concatenate([members[index] for index in sampled])
        draws.append(
            f1_score(
                truth[indices],
                candidate[indices],
                average="weighted",
                zero_division=0,
            )
            - f1_score(
                truth[indices],
                baseline[indices],
                average="weighted",
                zero_division=0,
            )
        )
    point = f1_score(truth, candidate, average="weighted", zero_division=0) - f1_score(
        truth,
        baseline,
        average="weighted",
        zero_division=0,
    )
    return float(point), np.asarray(draws, dtype=np.float64)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--iterations", type=int, default=2000)
    args = parser.parse_args()
    payloads = {
        (variant, seed): json.loads(
            _result(args.input, variant, seed).read_text(encoding="utf-8")
        )
        for variant in VARIANTS
        for seed in SEEDS
    }
    summary = {"runs_per_variant": 3, "metrics": {}, "paired_difference": {}}
    for variant in VARIANTS:
        summary["metrics"][variant] = {}
        for dataset in DATASETS:
            summary["metrics"][variant][dataset] = {}
            for metric in ("weighted_f1", "macro_f1", "accuracy"):
                values = np.asarray(
                    [
                        payloads[(variant, seed)]["test"][dataset][metric]
                        for seed in SEEDS
                    ],
                    dtype=np.float64,
                )
                summary["metrics"][variant][dataset][metric] = {
                    "mean": float(values.mean()),
                    "sample_std": float(values.std(ddof=1)),
                }
    all_draws = []
    all_points = []
    for dataset_index, dataset in enumerate(DATASETS):
        dataset_draws = []
        dataset_points = []
        for seed_index, seed in enumerate(SEEDS):
            truth, baseline, candidate, contexts = _load_aligned(
                _prediction(args.input, "v3_loss_only", seed, dataset),
                _prediction(args.input, "v3_ranked", seed, dataset),
            )
            point, draws = _paired_draws(
                truth,
                baseline,
                candidate,
                contexts,
                iterations=args.iterations,
                seed=20260726 + dataset_index * 10 + seed_index,
            )
            dataset_points.append(point)
            dataset_draws.append(draws)
            all_points.append(point)
            all_draws.append(draws)
        averaged = np.mean(np.stack(dataset_draws), axis=0)
        summary["paired_difference"][dataset] = {
            "ranked_minus_loss_only_weighted_f1": float(np.mean(dataset_points)),
            "ci95": [
                float(np.quantile(averaged, 0.025)),
                float(np.quantile(averaged, 0.975)),
            ],
            "bootstrap_unit": "context",
            "iterations": args.iterations,
        }
    bilingual_draws = np.mean(np.stack(all_draws), axis=0)
    summary["paired_difference"]["bilingual_average"] = {
        "ranked_minus_loss_only_weighted_f1": float(np.mean(all_points)),
        "ci95": [
            float(np.quantile(bilingual_draws, 0.025)),
            float(np.quantile(bilingual_draws, 0.975)),
        ],
        "bootstrap_unit": "context",
        "iterations": args.iterations,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

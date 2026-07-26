#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from bimer.v3_protocol import (
    select_classification_loss,
    select_gate_ranking_weight,
)

MODALITY_INDEX = {"whisper": 0, "audio_10db": 1, "video_50": 2}


def _validation(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))["validation"]


def _condition_metrics(result_path: Path) -> dict[str, dict]:
    root = result_path.parent / "validation_conditions"
    return {
        condition: _validation(root / f"{condition}.json")
        for condition in ("audio_10db", "video_50", "whisper")
    }


def _gate_delta(
    clean_root: Path,
    corrupted_json: Path,
    *,
    condition: str,
    iterations: int = 2000,
) -> dict[str, object]:
    clean_rows = []
    corrupted_rows = []
    clusters = []
    corrupted_root = corrupted_json.parent / f"{corrupted_json.stem}.predictions"
    modality = MODALITY_INDEX[condition]
    for dataset in ("meld", "emotiontalk"):
        with (
            np.load(
                clean_root / f"{dataset}.npz",
                allow_pickle=False,
            ) as clean,
            np.load(
                corrupted_root / f"{dataset}.npz",
                allow_pickle=False,
            ) as corrupted,
        ):
            clean_index = {
                sample_id: index for index, sample_id in enumerate(clean["sample_ids"].tolist())
            }
            if set(clean_index) != set(corrupted["sample_ids"].tolist()):
                raise ValueError(f"{condition}/{dataset} clean and corrupted sample ids differ")
            order = np.asarray(
                [clean_index[sample_id] for sample_id in corrupted["sample_ids"].tolist()]
            )
            if not np.array_equal(clean["truth"][order], corrupted["truth"]):
                raise ValueError(f"{condition}/{dataset} clean and corrupted labels differ")
            if not np.array_equal(
                clean["context_ids"][order].astype(str),
                corrupted["context_ids"].astype(str),
            ):
                raise ValueError(f"{condition}/{dataset} clean and corrupted context ids differ")
            valid = clean["modality_available"][order, modality].astype(bool) & corrupted[
                "modality_available"
            ][:, modality].astype(bool)
            if not valid.any():
                raise ValueError(f"{condition}/{dataset} has no paired available modality rows")
            clean_rows.append(clean["gates"][order, modality][valid])
            corrupted_rows.append(corrupted["gates"][:, modality][valid])
            clusters.extend(
                f"{dataset}:{context_id}" for context_id in corrupted["context_ids"][valid].tolist()
            )
    clean_values = np.concatenate(clean_rows)
    corrupted_values = np.concatenate(corrupted_rows)
    delta = corrupted_values - clean_values
    cluster_values = np.asarray(clusters, dtype=str)
    unique = np.unique(cluster_values)
    random = np.random.default_rng(42)
    bootstrap = []
    for _ in range(iterations):
        sampled = random.choice(unique, size=len(unique), replace=True)
        indices = np.concatenate([np.flatnonzero(cluster_values == cluster) for cluster in sampled])
        bootstrap.append(float(delta[indices].mean()))
    return {
        "mean": float(delta.mean()),
        "ci95": [
            float(np.quantile(bootstrap, 0.025)),
            float(np.quantile(bootstrap, 0.975)),
        ],
        "bootstrap_unit": "context",
        "iterations": iterations,
    }


def _ranking_payload(result_path: Path) -> dict:
    condition_root = result_path.parent / "validation_conditions"
    return {
        "clean": _validation(result_path),
        "perturbed": _condition_metrics(result_path),
        "gate_deltas": {
            "audio": _gate_delta(
                result_path.parent / "validation_predictions",
                condition_root / "audio_10db.json",
                condition="audio_10db",
            ),
            "vision": _gate_delta(
                result_path.parent / "validation_predictions",
                condition_root / "video_50.json",
                condition="video_50",
            ),
            "text": _gate_delta(
                result_path.parent / "validation_predictions",
                condition_root / "whisper.json",
                condition="whisper",
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    loss = subparsers.add_parser("loss")
    loss.add_argument("--baseline", required=True)
    loss.add_argument("--candidate", action="append", required=True)
    loss.add_argument("--output", required=True)
    ranking = subparsers.add_parser("ranking")
    ranking.add_argument("--baseline", required=True)
    ranking.add_argument("--candidate", action="append", required=True)
    ranking.add_argument("--output", required=True)
    args = parser.parse_args()

    candidates = {}
    for specification in args.candidate:
        name, path = specification.split("=", 1)
        candidates[name] = Path(path)
    if args.command == "loss":
        decision = select_classification_loss(
            baseline_name="weighted_ce",
            baseline=_validation(Path(args.baseline)),
            candidates={name: _validation(path) for name, path in candidates.items()},
        )
    else:
        baseline_path = Path(args.baseline)
        decision = select_gate_ranking_weight(
            baseline_clean=_validation(baseline_path),
            baseline_perturbed=_condition_metrics(baseline_path),
            candidates={float(name): _ranking_payload(path) for name, path in candidates.items()},
        )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(asdict(decision), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

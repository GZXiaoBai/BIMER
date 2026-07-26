#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from statistics import mean, stdev
from typing import Any

SEEDS = (42, 123, 2026)
DATASETS = ("meld", "emotiontalk")
FORMAL_VARIANTS = (
    "early_mlp",
    "early_context",
    "lagf_no_gates",
    "quality_lagf",
)
ABLATIONS = (
    "no_language",
    "no_gates",
    "no_context",
    "no_quality",
    "no_modality_dropout",
    "no_perturbation_training",
)
METRICS = ("weighted_f1", "macro_f1", "accuracy")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def _round(value: float) -> float:
    return round(float(value), 12)


def _result_path(root: Path, scope: str, variant: str, seed: int) -> Path:
    matches = sorted((root / scope / variant).glob(f"**/seed-{seed}/results.json"))
    if len(matches) != 1:
        raise ValueError(f"missing results for {scope}/{variant}/seed-{seed}: found {len(matches)}")
    return matches[0]


def _load_result(
    root: Path,
    *,
    scope: str,
    variant: str,
    seed: int,
) -> tuple[Path, dict[str, Any]]:
    path = _result_path(root, scope, variant, seed)
    payload = json.loads(path.read_text(encoding="utf-8"))
    config = payload.get("config", {})
    if config.get("seed") != seed:
        raise ValueError(f"seed mismatch in {path}")
    if config.get("learning_rate") != 0.0001:
        raise ValueError(f"learning-rate mismatch in {path}")
    if config.get("training_scope") != "joint":
        raise ValueError(f"training-scope mismatch in {path}")
    if config.get("evaluate_test") is not True:
        raise ValueError(f"test evaluation missing in {path}")
    if set(payload.get("test", {})) != set(DATASETS):
        raise ValueError(f"unexpected test datasets in {path}")
    return path, payload


def _per_seed_rows(
    root: Path,
    *,
    scope: str,
    variants: tuple[str, ...],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    per_class_rows: list[dict[str, Any]] = []
    for variant in variants:
        for seed in SEEDS:
            path, payload = _load_result(
                root,
                scope=scope,
                variant=variant,
                seed=seed,
            )
            dataset_rows: list[dict[str, Any]] = []
            for dataset in DATASETS:
                metrics = payload["test"][dataset]
                row = {
                    "scope": scope,
                    "variant": variant,
                    "seed": seed,
                    "dataset": dataset,
                    **{name: _round(metrics[name]) for name in METRICS},
                    "best_epoch": payload["history"]["best_epoch"],
                    "best_validation_score": _round(payload["history"]["best_score"]),
                    "result_path": str(path),
                }
                if not all(math.isfinite(row[name]) for name in METRICS):
                    raise ValueError(f"non-finite metric in {path}")
                rows.append(row)
                dataset_rows.append(row)
                for label, value in metrics["per_class_f1"].items():
                    per_class_rows.append(
                        {
                            "scope": scope,
                            "variant": variant,
                            "seed": seed,
                            "dataset": dataset,
                            "label": label,
                            "f1": _round(value),
                        }
                    )
            rows.append(
                {
                    "scope": scope,
                    "variant": variant,
                    "seed": seed,
                    "dataset": "bilingual_average",
                    **{name: _round(mean(row[name] for row in dataset_rows)) for name in METRICS},
                    "best_epoch": payload["history"]["best_epoch"],
                    "best_validation_score": _round(payload["history"]["best_score"]),
                    "result_path": str(path),
                }
            )
    return rows, per_class_rows


def _summary_rows(
    rows: list[dict[str, Any]],
    variants: tuple[str, ...],
) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    for variant in variants:
        for dataset in (*DATASETS, "bilingual_average"):
            selected = [
                row for row in rows if row["variant"] == variant and row["dataset"] == dataset
            ]
            if len(selected) != len(SEEDS):
                raise ValueError(f"missing results for summary {variant}/{dataset}")
            summary.append(
                {
                    "variant": variant,
                    "dataset": dataset,
                    "seed_count": len(selected),
                    **{
                        f"{metric}_mean": _round(mean(row[metric] for row in selected))
                        for metric in METRICS
                    },
                    **{
                        f"{metric}_std": _round(stdev(row[metric] for row in selected))
                        for metric in METRICS
                    },
                }
            )
    return summary


def _per_class_summary(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    keys = sorted({(row["scope"], row["variant"], row["dataset"], row["label"]) for row in rows})
    summary: list[dict[str, Any]] = []
    for scope, variant, dataset, label in keys:
        values = [
            row["f1"]
            for row in rows
            if (
                row["scope"],
                row["variant"],
                row["dataset"],
                row["label"],
            )
            == (scope, variant, dataset, label)
        ]
        summary.append(
            {
                "scope": scope,
                "variant": variant,
                "dataset": dataset,
                "label": label,
                "seed_count": len(values),
                "f1_mean": _round(mean(values)),
                "f1_std": _round(stdev(values)),
            }
        )
    return summary


def _add_ablation_deltas(
    ablation_summary: list[dict[str, Any]],
    formal_summary: list[dict[str, Any]],
) -> None:
    full = {row["dataset"]: row for row in formal_summary if row["variant"] == "quality_lagf"}
    for row in ablation_summary:
        for metric in METRICS:
            row[f"{metric}_delta_vs_full"] = _round(
                row[f"{metric}_mean"] - full[row["dataset"]][f"{metric}_mean"]
            )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty table: {path}")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(root: Path, output: Path) -> dict[str, Any]:
    formal_rows, formal_per_class = _per_seed_rows(
        root,
        scope="formal",
        variants=FORMAL_VARIANTS,
    )
    ablation_rows, ablation_per_class = _per_seed_rows(
        root,
        scope="ablations",
        variants=ABLATIONS,
    )
    formal_summary = _summary_rows(formal_rows, FORMAL_VARIANTS)
    ablation_summary = _summary_rows(ablation_rows, ABLATIONS)
    _add_ablation_deltas(ablation_summary, formal_summary)
    per_class_summary = _per_class_summary(formal_per_class + ablation_per_class)
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / "formal_per_seed.csv", formal_rows)
    _write_csv(output / "formal_summary.csv", formal_summary)
    _write_csv(output / "ablation_per_seed.csv", ablation_rows)
    _write_csv(output / "ablation_summary.csv", ablation_summary)
    _write_csv(output / "per_class_f1_summary.csv", per_class_summary)
    payload = {
        "methodology": {
            "seeds": list(SEEDS),
            "standard_deviation_ddof": 1,
            "bilingual_average": ("Arithmetic mean of MELD and EmotionTalk metrics per seed"),
            "ablation_delta": "Ablation mean minus full-model mean",
        },
        "validation": {
            "formal_result_count": len(FORMAL_VARIANTS) * len(SEEDS),
            "ablation_result_count": len(ABLATIONS) * len(SEEDS),
            "datasets": list(DATASETS),
        },
        "formal_summary": formal_summary,
        "ablation_summary": ablation_summary,
        "per_class_f1_summary": per_class_summary,
    }
    (output / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return payload


def main() -> int:
    args = parse_args()
    try:
        summarize(args.input, args.output)
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
# ruff: noqa: E402
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bimer.labels import EMOTION_LABELS
from bimer.metrics import (
    classification_metrics,
    paired_cluster_bootstrap_weighted_f1_delta,
)
from bimer.v4_analysis import (
    analyze_context_gates,
    context_stratified_metrics,
    ensemble_predictions,
    prototype_geometry,
    should_enable_ensemble,
)

DATASETS = ("meld", "emotiontalk")
SEEDS = (42, 123, 2026)
VARIANTS = ("full", "no_context_gate", "no_prototype", "neither")
MODEL = "adaptive_context_prototype"


def _run_root(formal_root: Path, variant: str, seed: int) -> Path:
    return formal_root / variant / MODEL / "joint" / f"seed-{seed}"


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return {name: payload[name].copy() for name in payload.files}


def _aggregate(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)),
    }


def _variant_summary(formal_root: Path, variant: str) -> dict[str, object]:
    payloads = [
        json.loads(
            (_run_root(formal_root, variant, seed) / "results.json").read_text(encoding="utf-8")
        )
        for seed in SEEDS
    ]
    summary: dict[str, object] = {}
    for dataset in DATASETS:
        summary[dataset] = {
            metric: _aggregate(
                [float(payload["validation"][dataset][metric]) for payload in payloads]
            )
            for metric in ("weighted_f1", "macro_f1", "accuracy")
        }
        summary[dataset]["per_class_f1"] = {
            label: _aggregate(
                [
                    float(payload["validation"][dataset]["per_class_f1"][label])
                    for payload in payloads
                ]
            )
            for label in EMOTION_LABELS
        }
    summary["bilingual_average"] = {
        metric: _aggregate(
            [
                float(np.mean([payload["validation"][dataset][metric] for dataset in DATASETS]))
                for payload in payloads
            ]
        )
        for metric in ("weighted_f1", "macro_f1", "accuracy")
    }
    return summary


def _ensemble(formal_root: Path, output_root: Path) -> tuple[dict, dict[str, dict]]:
    reports: dict[str, dict] = {}
    arrays: dict[str, dict] = {}
    output_root.mkdir(parents=True, exist_ok=True)
    for dataset in DATASETS:
        seed_paths = [
            _run_root(formal_root, "full", seed) / "validation_predictions" / f"{dataset}.npz"
            for seed in SEEDS
        ]
        payload = ensemble_predictions(seed_paths)
        metrics = classification_metrics(
            payload["truth"],
            payload["prediction"],
            label_names=EMOTION_LABELS,
        )
        reports[dataset] = metrics
        arrays[dataset] = payload
        np.savez_compressed(output_root / f"{dataset}.npz", **payload)
    return {
        "probability_average": True,
        "seeds": list(SEEDS),
        "validation": reports,
    }, arrays


def _bilingual(report: dict[str, dict], metric: str) -> float:
    return float(np.mean([report[dataset][metric] for dataset in DATASETS]))


def _formal_stability(
    variants: dict[str, object],
    baseline_path: Path | None,
) -> tuple[bool, dict[str, object]]:
    if baseline_path is None:
        return False, {"reason": "baseline_not_provided"}
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))["validation"]
    full = variants["full"]
    weighted_gain = float(
        np.mean(
            [
                full[dataset]["weighted_f1"]["mean"] - baseline[dataset]["weighted_f1"]
                for dataset in DATASETS
            ]
        )
    )
    macro_gain = float(
        np.mean(
            [
                full[dataset]["macro_f1"]["mean"] - baseline[dataset]["macro_f1"]
                for dataset in DATASETS
            ]
        )
    )
    worst_dataset_delta = float(
        min(
            full[dataset]["weighted_f1"]["mean"] - baseline[dataset]["weighted_f1"]
            for dataset in DATASETS
        )
    )
    minority = ("fear", "disgust", "sadness")
    minority_gain = float(
        np.mean(
            [
                full[dataset]["per_class_f1"][label]["mean"]
                - baseline[dataset]["per_class_f1"][label]
                for dataset in DATASETS
                for label in minority
            ]
        )
    )
    diagnostics = {
        "bilingual_weighted_f1_gain": weighted_gain,
        "bilingual_macro_f1_gain": macro_gain,
        "worst_dataset_weighted_f1_delta": worst_dataset_delta,
        "minority_f1_gain": minority_gain,
        "thresholds": {
            "bilingual_weighted_f1_gain": 0.010,
            "bilingual_macro_f1_gain": 0.008,
            "worst_dataset_weighted_f1_delta": -0.003,
            "minority_f1_gain": 0.015,
        },
    }
    stable = (
        weighted_gain >= 0.010
        and macro_gain >= 0.008
        and worst_dataset_delta >= -0.003
        and minority_gain >= 0.015
    )
    return stable, diagnostics


def summarize(
    *,
    formal_root: Path,
    output: Path,
    bootstrap_iterations: int,
    baseline_path: Path | None = None,
) -> Path:
    variants = {variant: _variant_summary(formal_root, variant) for variant in VARIANTS}
    ensemble, ensemble_arrays = _ensemble(
        formal_root,
        output.parent / "v4-ensemble-validation",
    )
    seed42_metrics = json.loads(
        (_run_root(formal_root, "full", 42) / "results.json").read_text(encoding="utf-8")
    )["validation"]
    ensemble["enable_for_validation"] = should_enable_ensemble(
        single={
            "weighted_f1": _bilingual(seed42_metrics, "weighted_f1"),
            "macro_f1": _bilingual(seed42_metrics, "macro_f1"),
        },
        ensemble={
            "weighted_f1": _bilingual(ensemble["validation"], "weighted_f1"),
            "macro_f1": _bilingual(ensemble["validation"], "macro_f1"),
        },
    )

    context_strata: dict[str, object] = {}
    context_gates: dict[str, object] = {}
    geometry: dict[str, object] = {}
    for dataset in DATASETS:
        seed42 = _load_npz(
            _run_root(formal_root, "full", 42) / "validation_predictions" / f"{dataset}.npz"
        )
        context_strata[dataset] = context_stratified_metrics(
            ensemble_arrays[dataset]["truth"],
            ensemble_arrays[dataset]["prediction"],
            seed42["context_lengths"],
            label_names=EMOTION_LABELS,
        )
        context_gates[dataset] = {
            str(seed): analyze_context_gates(
                (
                    prediction := _load_npz(
                        _run_root(formal_root, "full", seed)
                        / "validation_predictions"
                        / f"{dataset}.npz"
                    )
                )["context_gates"],
                prediction["context_lengths"],
                prediction["local_prediction"],
                prediction["fixed_context_prediction"],
            )
            for seed in SEEDS
        }
        geometry[dataset] = {
            str(seed): prototype_geometry(
                (
                    prediction := _load_npz(
                        _run_root(formal_root, "full", seed)
                        / "validation_predictions"
                        / f"{dataset}.npz"
                    )
                )["representations"],
                prediction["truth"],
                num_classes=len(EMOTION_LABELS),
            )
            for seed in SEEDS
        }

    paired: dict[str, object] = {}
    for variant in VARIANTS[1:]:
        paired[variant] = {}
        for dataset in DATASETS:
            paired[variant][dataset] = {}
            for seed in SEEDS:
                full = _load_npz(
                    _run_root(formal_root, "full", seed)
                    / "validation_predictions"
                    / f"{dataset}.npz"
                )
                ablation = _load_npz(
                    _run_root(formal_root, variant, seed)
                    / "validation_predictions"
                    / f"{dataset}.npz"
                )
                if not np.array_equal(full["sample_ids"], ablation["sample_ids"]):
                    raise ValueError(f"{variant} seed {seed} predictions are not paired")
                interval = paired_cluster_bootstrap_weighted_f1_delta(
                    full["truth"],
                    ablation["prediction"],
                    full["prediction"],
                    full["context_ids"],
                    iterations=bootstrap_iterations,
                    seed=seed,
                )
                paired[variant][dataset][str(seed)] = {
                    "candidate_minus_ablation_ci95": list(interval),
                }

    formal_stable, stability = _formal_stability(variants, baseline_path)
    payload = {
        "evidence_scope": "validation_only",
        "test_set_used": False,
        "seeds": list(SEEDS),
        "std_ddof": 1,
        "variants": variants,
        "ensemble": ensemble,
        "evidence": {
            "context_strata": context_strata,
            "context_gate_relationship": context_gates,
            "prototype_geometry": geometry,
        },
        "paired_bootstrap": paired,
        "bootstrap_iterations": bootstrap_iterations,
        "bootstrap_unit": "context_id",
        "formal_stable": formal_stable,
        "formal_stability": stability,
    }
    _atomic_json(output, payload)
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--formal-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    parser.add_argument("--baseline", type=Path)
    args = parser.parse_args()
    print(
        summarize(
            formal_root=args.formal_root,
            output=args.output,
            bootstrap_iterations=args.bootstrap_iterations,
            baseline_path=args.baseline,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

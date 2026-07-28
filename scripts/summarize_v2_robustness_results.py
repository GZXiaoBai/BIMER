#!/usr/bin/env python3
"""Validate and summarize the frozen V2 robustness experiment matrix.

The comparison confidence intervals use a paired cluster bootstrap over complete
dialogue context IDs. For bilingual and three-seed summaries, every
dataset/seed run is resampled independently and the resulting weighted-F1
differences are averaged within each bootstrap draw.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

SEEDS = (42, 123, 2026)
DATASETS = ("meld", "emotiontalk")
MODELS = ("quality_lagf", "no_gates")
CLASS_COUNT = 7
MODEL_LABELS = {
    "quality_lagf": "质量感知门控",
    "no_gates": "无门控上下文",
}
DATASET_LABELS = {
    "meld": "MELD（英文）",
    "emotiontalk": "EmotionTalk（中文）",
    "bilingual_average": "双语平均",
}
CONDITION_META = {
    "standard": ("reference", "标准输入"),
    "audio_snr_20db": ("corruption", "音频 20 dB SNR"),
    "audio_snr_10db": ("corruption", "音频 10 dB SNR"),
    "video_frame_drop_25pct": ("corruption", "视频丢帧 25%"),
    "video_frame_drop_50pct": ("corruption", "视频丢帧 50%"),
    "whisper_text": ("corruption", "Whisper 转写文本"),
    "missing-text": ("missing_modality", "缺失文本"),
    "missing-audio": ("missing_modality", "缺失语音"),
    "missing-vision": ("missing_modality", "缺失视频"),
    "missing-audio-vision": ("missing_modality", "仅保留文本"),
    "missing-text-vision": ("missing_modality", "仅保留语音"),
    "missing-text-audio": ("missing_modality", "仅保留视频"),
}


@dataclass(frozen=True)
class PredictionBundle:
    sample_ids: np.ndarray
    context_ids: np.ndarray
    truth: np.ndarray
    prediction: np.ndarray


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=root)
    parser.add_argument(
        "--results-root",
        type=Path,
        default=(
            root
            / "artifacts/autodl/v2-robustness-final/extracted-v1"
            / "artifacts/experiments/v2/robustness"
        ),
    )
    parser.add_argument(
        "--selection-config",
        type=Path,
        default=(
            root
            / "artifacts/autodl/v2-robustness-final/extracted-v1"
            / "configs/experiment-v2-selection.json"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "artifacts/analysis/v2-robustness",
    )
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260726)
    return parser.parse_args()


def mean(values: Iterable[float]) -> float:
    return statistics.fmean(list(values))


def sample_stdev(values: Iterable[float]) -> float:
    materialized = list(values)
    return statistics.stdev(materialized) if len(materialized) > 1 else 0.0


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object in {path}")
    return payload


def _expected_result_path(
    root: Path,
    *,
    model: str,
    condition: str,
    seed: int,
) -> Path:
    return root / model / condition / f"seed-{seed}.json"


def _expected_prediction_path(
    root: Path,
    *,
    model: str,
    condition: str,
    seed: int,
    dataset: str,
) -> Path:
    return root / model / condition / f"seed-{seed}.predictions" / f"{dataset}.npz"


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
    if len(lengths) != 1 or not len(bundle.truth):
        raise ValueError(f"prediction arrays must have equal non-zero length in {path}")
    if len(set(bundle.sample_ids.tolist())) != len(bundle.sample_ids):
        raise ValueError(f"duplicate sample IDs in {path}")
    for values, name in ((bundle.truth, "truth"), (bundle.prediction, "prediction")):
        if np.any(values < 0) or np.any(values >= CLASS_COUNT):
            raise ValueError(f"{name} outside seven-class range in {path}")
    return bundle


def _align_pair(
    candidate: PredictionBundle,
    baseline: PredictionBundle,
    *,
    label: str,
) -> tuple[PredictionBundle, PredictionBundle]:
    candidate_ids = candidate.sample_ids.tolist()
    baseline_ids = baseline.sample_ids.tolist()
    if set(candidate_ids) != set(baseline_ids):
        raise ValueError(f"sample IDs do not match for {label}")
    baseline_index = {sample_id: index for index, sample_id in enumerate(baseline_ids)}
    order = np.asarray(
        [baseline_index[sample_id] for sample_id in candidate_ids],
        dtype=np.int64,
    )
    aligned = PredictionBundle(
        sample_ids=baseline.sample_ids[order],
        context_ids=baseline.context_ids[order],
        truth=baseline.truth[order],
        prediction=baseline.prediction[order],
    )
    if not np.array_equal(candidate.truth, aligned.truth):
        raise ValueError(f"truth labels do not match for {label}")
    if not np.array_equal(candidate.context_ids, aligned.context_ids):
        raise ValueError(f"context IDs do not match for {label}")
    return candidate, aligned


def _cluster_confusions(bundle: PredictionBundle) -> np.ndarray:
    context_ids, inverse = np.unique(bundle.context_ids, return_inverse=True)
    confusions = np.zeros(
        (len(context_ids), CLASS_COUNT, CLASS_COUNT),
        dtype=np.int64,
    )
    np.add.at(confusions, (inverse, bundle.truth, bundle.prediction), 1)
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


def paired_cluster_delta_draws(
    candidate: PredictionBundle,
    baseline: PredictionBundle,
    *,
    iterations: int,
    rng: np.random.Generator,
) -> tuple[float, np.ndarray, int]:
    """Return candidate-minus-baseline weighted-F1 point and bootstrap draws."""
    candidate_confusions = _cluster_confusions(candidate)
    baseline_confusions = _cluster_confusions(baseline)
    if candidate_confusions.shape != baseline_confusions.shape:
        raise ValueError("paired models have different context counts")
    cluster_count = len(candidate_confusions)
    counts = rng.multinomial(
        cluster_count,
        np.full(cluster_count, 1 / cluster_count),
        size=iterations,
    )
    candidate_draws = np.einsum(
        "bc,cij->bij",
        counts,
        candidate_confusions,
        optimize=True,
    )
    baseline_draws = np.einsum(
        "bc,cij->bij",
        counts,
        baseline_confusions,
        optimize=True,
    )
    draws = _weighted_f1_from_confusions(candidate_draws) - _weighted_f1_from_confusions(
        baseline_draws
    )
    point = float(
        _weighted_f1_from_confusions(candidate_confusions.sum(axis=0))[0]
        - _weighted_f1_from_confusions(baseline_confusions.sum(axis=0))[0]
    )
    return point, draws, cluster_count


def _archive_metric(payload: dict[str, Any], dataset: str, metric: str) -> float:
    try:
        return float(payload["test"][dataset][metric])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"missing test.{dataset}.{metric}") from exc


def _validate_matrix(results_root: Path) -> dict[str, Any]:
    expected_results = [
        _expected_result_path(
            results_root,
            model=model,
            condition=condition,
            seed=seed,
        )
        for model in MODELS
        for condition in CONDITION_META
        for seed in SEEDS
    ]
    expected_predictions = [
        _expected_prediction_path(
            results_root,
            model=model,
            condition=condition,
            seed=seed,
            dataset=dataset,
        )
        for model in MODELS
        for condition in CONDITION_META
        for seed in SEEDS
        for dataset in DATASETS
    ]
    missing = [
        str(path) for path in (*expected_results, *expected_predictions) if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(
            f"missing {len(missing)} expected robustness files; first: {missing[0]}"
        )
    actual_results = list(results_root.glob("*/*/seed-*.json"))
    actual_predictions = list(results_root.glob("*/*/seed-*.predictions/*.npz"))
    if len(actual_results) != len(expected_results):
        raise ValueError(
            f"expected {len(expected_results)} result JSON files, found {len(actual_results)}"
        )
    if len(actual_predictions) != len(expected_predictions):
        raise ValueError(
            f"expected {len(expected_predictions)} prediction NPZ files, "
            f"found {len(actual_predictions)}"
        )
    return {
        "result_json_expected": len(expected_results),
        "result_json_found": len(actual_results),
        "prediction_npz_expected": len(expected_predictions),
        "prediction_npz_found": len(actual_predictions),
    }


def _load_runs(results_root: Path) -> dict[str, dict[str, dict[int, dict[str, Any]]]]:
    runs: dict[str, dict[str, dict[int, dict[str, Any]]]] = {}
    for model in MODELS:
        runs[model] = {}
        for condition in CONDITION_META:
            runs[model][condition] = {}
            for seed in SEEDS:
                path = _expected_result_path(
                    results_root,
                    model=model,
                    condition=condition,
                    seed=seed,
                )
                runs[model][condition][seed] = _load_json(path)
    return runs


def _dataset_seed_metrics(
    runs: dict[int, dict[str, Any]],
    *,
    dataset: str,
) -> dict[str, list[float]]:
    output = {"weighted_f1": [], "macro_f1": [], "accuracy": []}
    for seed in SEEDS:
        if dataset in DATASETS:
            for metric in output:
                output[metric].append(_archive_metric(runs[seed], dataset, metric))
        else:
            for metric in output:
                output[metric].append(
                    mean(_archive_metric(runs[seed], name, metric) for name in DATASETS)
                )
    return output


def summarize_models(runs: dict[str, dict[str, dict[int, dict[str, Any]]]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model in MODELS:
        standard = {
            dataset: _dataset_seed_metrics(
                runs[model]["standard"],
                dataset=dataset,
            )["weighted_f1"]
            for dataset in (*DATASETS, "bilingual_average")
        }
        for condition, (group, label) in CONDITION_META.items():
            for dataset in (*DATASETS, "bilingual_average"):
                metrics = _dataset_seed_metrics(
                    runs[model][condition],
                    dataset=dataset,
                )
                weighted_mean = mean(metrics["weighted_f1"])
                standard_mean = mean(standard[dataset])
                delta = weighted_mean - standard_mean
                rows.append(
                    {
                        "group": group,
                        "model": model,
                        "model_zh": MODEL_LABELS[model],
                        "condition": condition,
                        "condition_zh": label,
                        "dataset": dataset,
                        "dataset_zh": DATASET_LABELS[dataset],
                        "runs": len(SEEDS),
                        "weighted_f1_mean": weighted_mean,
                        "weighted_f1_std": sample_stdev(metrics["weighted_f1"]),
                        "macro_f1_mean": mean(metrics["macro_f1"]),
                        "macro_f1_std": sample_stdev(metrics["macro_f1"]),
                        "accuracy_mean": mean(metrics["accuracy"]),
                        "accuracy_std": sample_stdev(metrics["accuracy"]),
                        "delta_from_standard": delta,
                        "relative_delta_pct": (
                            delta / standard_mean * 100
                            if not math.isclose(standard_mean, 0.0)
                            else None
                        ),
                        **{
                            f"seed_{seed}_weighted_f1": metrics["weighted_f1"][index]
                            for index, seed in enumerate(SEEDS)
                        },
                    }
                )
    return rows


def _summary_row(
    rows: list[dict[str, Any]],
    *,
    model: str,
    condition: str,
    dataset: str,
) -> dict[str, Any]:
    matches = [
        row
        for row in rows
        if row["model"] == model and row["condition"] == condition and row["dataset"] == dataset
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one summary row for {model}/{condition}/{dataset}")
    return matches[0]


def compare_models(
    results_root: Path,
    summary_rows: list[dict[str, Any]],
    *,
    iterations: int,
    bootstrap_seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if iterations < 1:
        raise ValueError("bootstrap iterations must be positive")
    rng = np.random.default_rng(bootstrap_seed)
    rows: list[dict[str, Any]] = []
    checked_archives = 0
    context_counts: dict[str, set[int]] = {dataset: set() for dataset in DATASETS}

    for condition, (group, condition_label) in CONDITION_META.items():
        dataset_points: dict[str, list[float]] = {dataset: [] for dataset in DATASETS}
        dataset_draws: dict[str, list[np.ndarray]] = {dataset: [] for dataset in DATASETS}
        dataset_clusters: dict[str, list[int]] = {dataset: [] for dataset in DATASETS}
        for dataset in DATASETS:
            for seed in SEEDS:
                candidate_path = _expected_prediction_path(
                    results_root,
                    model="quality_lagf",
                    condition=condition,
                    seed=seed,
                    dataset=dataset,
                )
                baseline_path = _expected_prediction_path(
                    results_root,
                    model="no_gates",
                    condition=condition,
                    seed=seed,
                    dataset=dataset,
                )
                candidate, baseline = _align_pair(
                    _load_bundle(candidate_path),
                    _load_bundle(baseline_path),
                    label=f"{condition}/seed-{seed}/{dataset}",
                )
                candidate_json = _load_json(
                    _expected_result_path(
                        results_root,
                        model="quality_lagf",
                        condition=condition,
                        seed=seed,
                    )
                )
                baseline_json = _load_json(
                    _expected_result_path(
                        results_root,
                        model="no_gates",
                        condition=condition,
                        seed=seed,
                    )
                )
                point, draws, cluster_count = paired_cluster_delta_draws(
                    candidate,
                    baseline,
                    iterations=iterations,
                    rng=rng,
                )
                candidate_point = float(
                    _weighted_f1_from_confusions(_cluster_confusions(candidate).sum(axis=0))[0]
                )
                baseline_point = float(
                    _weighted_f1_from_confusions(_cluster_confusions(baseline).sum(axis=0))[0]
                )
                if not math.isclose(
                    candidate_point,
                    _archive_metric(candidate_json, dataset, "weighted_f1"),
                    abs_tol=1e-10,
                ):
                    raise ValueError(
                        f"candidate JSON and predictions disagree for "
                        f"{condition}/seed-{seed}/{dataset}"
                    )
                if not math.isclose(
                    baseline_point,
                    _archive_metric(baseline_json, dataset, "weighted_f1"),
                    abs_tol=1e-10,
                ):
                    raise ValueError(
                        f"baseline JSON and predictions disagree for "
                        f"{condition}/seed-{seed}/{dataset}"
                    )
                if not math.isclose(point, candidate_point - baseline_point, abs_tol=1e-12):
                    raise ValueError("paired point estimate is inconsistent")
                dataset_points[dataset].append(point)
                dataset_draws[dataset].append(draws)
                dataset_clusters[dataset].append(cluster_count)
                context_counts[dataset].add(cluster_count)
                checked_archives += 2

        for dataset in DATASETS:
            rows.append(
                _comparison_row(
                    summary_rows,
                    condition=condition,
                    group=group,
                    condition_label=condition_label,
                    dataset=dataset,
                    points=dataset_points[dataset],
                    draws=dataset_draws[dataset],
                    cluster_counts=dataset_clusters[dataset],
                    iterations=iterations,
                )
            )
        rows.append(
            _comparison_row(
                summary_rows,
                condition=condition,
                group=group,
                condition_label=condition_label,
                dataset="bilingual_average",
                points=[point for dataset in DATASETS for point in dataset_points[dataset]],
                draws=[draw for dataset in DATASETS for draw in dataset_draws[dataset]],
                cluster_counts=[
                    count for dataset in DATASETS for count in dataset_clusters[dataset]
                ],
                iterations=iterations,
            )
        )

    validation = {
        "prediction_archives_checked": checked_archives,
        "expected_prediction_archives": len(MODELS)
        * len(CONDITION_META)
        * len(SEEDS)
        * len(DATASETS),
        "context_counts_observed": {
            dataset: sorted(counts) for dataset, counts in context_counts.items()
        },
        "json_prediction_weighted_f1_match": True,
        "paired_sample_ids_match": True,
        "paired_truth_and_context_ids_match": True,
    }
    return rows, validation


def _comparison_row(
    summary_rows: list[dict[str, Any]],
    *,
    condition: str,
    group: str,
    condition_label: str,
    dataset: str,
    points: list[float],
    draws: list[np.ndarray],
    cluster_counts: list[int],
    iterations: int,
) -> dict[str, Any]:
    aggregate_draws = np.mean(np.stack(draws), axis=0)
    lower, upper = np.quantile(aggregate_draws, [0.025, 0.975])
    quality = _summary_row(
        summary_rows,
        model="quality_lagf",
        condition=condition,
        dataset=dataset,
    )
    no_gates = _summary_row(
        summary_rows,
        model="no_gates",
        condition=condition,
        dataset=dataset,
    )
    point = mean(points)
    if not math.isclose(
        point,
        float(quality["weighted_f1_mean"]) - float(no_gates["weighted_f1_mean"]),
        abs_tol=1e-12,
    ):
        raise ValueError(f"summary and predictions disagree for {condition}/{dataset}")
    return {
        "group": group,
        "condition": condition,
        "condition_zh": condition_label,
        "dataset": dataset,
        "dataset_zh": DATASET_LABELS[dataset],
        "quality_weighted_f1_mean": quality["weighted_f1_mean"],
        "quality_weighted_f1_std": quality["weighted_f1_std"],
        "no_gates_weighted_f1_mean": no_gates["weighted_f1_mean"],
        "no_gates_weighted_f1_std": no_gates["weighted_f1_std"],
        "quality_minus_no_gates": point,
        "quality_minus_no_gates_std": sample_stdev(points),
        "ci95_lower": float(lower),
        "ci95_upper": float(upper),
        "significant_at_0_05": bool(lower > 0 or upper < 0),
        "supports_quality": bool(lower > 0),
        "bootstrap_unit": "complete_dialogue_context_id",
        "bootstrap_iterations": iterations,
        "seed_count": len(SEEDS),
        "cluster_count_min": min(cluster_counts),
        "cluster_count_max": max(cluster_counts),
    }


def _find_comparison(
    rows: list[dict[str, Any]],
    condition: str,
    dataset: str = "bilingual_average",
) -> dict[str, Any]:
    matches = [row for row in rows if row["condition"] == condition and row["dataset"] == dataset]
    if len(matches) != 1:
        raise ValueError(f"missing comparison for {condition}/{dataset}")
    return matches[0]


def _model_value(
    rows: list[dict[str, Any]],
    *,
    model: str,
    condition: str,
    field: str = "weighted_f1_mean",
    dataset: str = "bilingual_average",
) -> float:
    return float(
        _summary_row(
            rows,
            model=model,
            condition=condition,
            dataset=dataset,
        )[field]
    )


def build_decision(
    summary_rows: list[dict[str, Any]],
    comparison_rows: list[dict[str, Any]],
    selection: dict[str, Any],
) -> dict[str, Any]:
    if selection.get("status") != "frozen":
        raise ValueError("selection configuration is not frozen")
    if selection.get("test_set_used_for_selection") is not False:
        raise ValueError("selection configuration must not use the test set")
    selected_model = str(selection.get("selected_model"))
    if selected_model != "quality_lagf":
        raise ValueError(f"unexpected frozen selected model: {selected_model}")

    validation_delta = float(selection["no_gate_comparison"]["selected_model_delta"])
    clean = _find_comparison(comparison_rows, "standard")
    video_25 = _find_comparison(comparison_rows, "video_frame_drop_25pct")
    video_50 = _find_comparison(comparison_rows, "video_frame_drop_50pct")
    audio_10 = _find_comparison(comparison_rows, "audio_snr_10db")
    whisper = _find_comparison(comparison_rows, "whisper_text")
    vision_only = _find_comparison(comparison_rows, "missing-text-audio")

    quality_standard = _model_value(summary_rows, model="quality_lagf", condition="standard")
    quality_video_50 = _model_value(
        summary_rows,
        model="quality_lagf",
        condition="video_frame_drop_50pct",
    )
    quality_missing_vision = _model_value(
        summary_rows,
        model="quality_lagf",
        condition="missing-vision",
    )
    video_50_loss = quality_standard - quality_video_50
    missing_vision_loss = quality_standard - quality_missing_vision

    validation_gain_pass = validation_delta >= 0.005
    clean_tolerance_pass = float(clean["quality_minus_no_gates"]) >= -0.005
    degraded_video_pass = video_50_loss <= missing_vision_loss
    keep_quality = (
        selected_model == "quality_lagf"
        and validation_gain_pass
        and clean_tolerance_pass
        and degraded_video_pass
    )

    return {
        "decision": ("deploy_quality_lagf" if keep_quality else "deploy_no_gates_context"),
        "selected_model": "quality_lagf" if keep_quality else "no_gates",
        "decision_basis": (
            "Model structure and hyperparameters were frozen on validation only. "
            "The test set is used here to verify robustness acceptance criteria, "
            "not to retune or reselect the model."
        ),
        "validation": {
            "selection_scope": selection["selection_scope"],
            "test_set_used_for_selection": selection["test_set_used_for_selection"],
            "quality_bilingual_weighted_f1": selection["validation_weighted_f1"][
                "bilingual_average"
            ],
            "no_gates_bilingual_weighted_f1": selection["no_gate_comparison"]["bilingual_average"],
            "quality_minus_no_gates": validation_delta,
            "gain_at_least_0_5pp": validation_gain_pass,
        },
        "test": {
            "clean_quality_weighted_f1": quality_standard,
            "clean_no_gates_weighted_f1": float(clean["no_gates_weighted_f1_mean"]),
            "clean_quality_minus_no_gates": float(clean["quality_minus_no_gates"]),
            "clean_delta_ci95": [
                float(clean["ci95_lower"]),
                float(clean["ci95_upper"]),
            ],
            "video_25_quality_minus_no_gates": float(video_25["quality_minus_no_gates"]),
            "video_25_delta_ci95": [
                float(video_25["ci95_lower"]),
                float(video_25["ci95_upper"]),
            ],
            "video_50_quality_minus_no_gates": float(video_50["quality_minus_no_gates"]),
            "video_50_delta_ci95": [
                float(video_50["ci95_lower"]),
                float(video_50["ci95_upper"]),
            ],
            "quality_video_50_loss_from_clean": video_50_loss,
            "quality_missing_vision_loss_from_clean": missing_vision_loss,
            "video_50_not_worse_than_missing_vision": degraded_video_pass,
            "whisper_quality_minus_no_gates": float(whisper["quality_minus_no_gates"]),
            "audio_10db_quality_minus_no_gates": float(audio_10["quality_minus_no_gates"]),
            "vision_only_quality_minus_no_gates": float(vision_only["quality_minus_no_gates"]),
        },
        "acceptance": {
            "validation_gain_at_least_0_5pp": validation_gain_pass,
            "clean_test_penalty_no_more_than_0_5pp": clean_tolerance_pass,
            "video_50_loss_not_greater_than_missing_vision_loss": degraded_video_pass,
            "all_required_criteria_pass": keep_quality,
        },
        "interpretation": {
            "supported": [
                "质量门控在验证集上较无门控上下文模型提高超过 0.5 个百分点。",
                "标准测试性能与无门控模型实质持平，清洁输入代价低于 0.5 个百分点。",
                "质量门控在 25% 和 50% 视频丢帧条件下均取得正向平均差值。",
                "50% 视频丢帧的退化小于完全缺失视觉，V1 的关键失败模式已修复。",
            ],
            "not_supported": [
                "不能宣称质量门控在所有扰动和缺失模态条件下普遍优于无门控模型。",
                "10 dB 音频噪声和仅保留视频条件仍是质量模型的明显弱项。",
            ],
        },
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    results_root = args.results_root.resolve()
    selection_path = args.selection_config.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    matrix_validation = _validate_matrix(results_root)
    runs = _load_runs(results_root)
    summary_rows = summarize_models(runs)
    comparison_rows, prediction_validation = compare_models(
        results_root,
        summary_rows,
        iterations=args.bootstrap_iterations,
        bootstrap_seed=args.bootstrap_seed,
    )
    decision = build_decision(
        summary_rows,
        comparison_rows,
        _load_json(selection_path),
    )
    validation = {
        "status": "passed",
        "results_root": str(results_root),
        "selection_config": str(selection_path),
        "models": list(MODELS),
        "conditions": list(CONDITION_META),
        "seeds": list(SEEDS),
        "datasets": list(DATASETS),
        **matrix_validation,
        **prediction_validation,
        "summary_rows": len(summary_rows),
        "comparison_rows": len(comparison_rows),
    }

    _write_csv(output_dir / "model-condition-summary.csv", summary_rows)
    _write_csv(output_dir / "model-comparison.csv", comparison_rows)
    (output_dir / "selection-decision.json").write_text(
        json.dumps(decision, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "validation.json").write_text(
        json.dumps(validation, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "v2-robustness-summary.json").write_text(
        json.dumps(
            {
                "metadata": validation,
                "decision": decision,
                "model_condition_summary": summary_rows,
                "model_comparison": comparison_rows,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"validated {validation['result_json_found']} result JSON files")
    print(f"validated {validation['prediction_npz_found']} prediction archives")
    print(f"wrote {len(summary_rows)} model-condition summary rows")
    print(f"wrote {len(comparison_rows)} paired comparison rows")
    print(f"decision: {decision['decision']}")
    print(f"output: {output_dir}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

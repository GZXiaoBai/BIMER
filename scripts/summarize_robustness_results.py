#!/usr/bin/env python3
"""Summarize bilingual robustness experiments into thesis-ready tables.

The script intentionally uses only Python's standard library so it can be run on
the local Mac without recreating the training environment.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from pathlib import Path
from typing import Any, Iterable

SEEDS = (42, 123, 2026)
DATASETS = ("meld", "emotiontalk")
CLASSES = ("neutral", "joy", "sadness", "anger", "surprise", "fear", "disgust")

CONDITION_META = {
    "standard": ("reference", "标准输入"),
    "audio_snr_20db": ("corruption", "音频 20 dB SNR"),
    "audio_snr_10db": ("corruption", "音频 10 dB SNR"),
    "video_frame_drop_25pct": ("corruption", "视频丢帧 25%"),
    "video_frame_drop_50pct": ("corruption", "视频丢帧 50%"),
    "whisper_text": ("corruption", "Whisper 转写文本"),
    "missing-audio": ("missing_modality", "缺失语音"),
    "missing-vision": ("missing_modality", "缺失视频"),
    "missing-text": ("missing_modality", "缺失文本"),
    "missing-audio-vision": ("missing_modality", "仅保留文本"),
    "missing-text-vision": ("missing_modality", "仅保留语音"),
    "missing-text-audio": ("missing_modality", "仅保留视频"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to artifacts/analysis/robustness under the project root.",
    )
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def mean(values: Iterable[float]) -> float:
    materialized = list(values)
    return statistics.fmean(materialized)


def sample_stdev(values: Iterable[float]) -> float:
    materialized = list(values)
    return statistics.stdev(materialized) if len(materialized) > 1 else 0.0


def result_path(root: Path, condition: str, seed: int) -> Path:
    if condition == "standard":
        return root / "artifacts/experiments/joint/full/lagf/joint" / f"seed-{seed}/results.json"
    if condition.startswith("missing-"):
        return (
            root
            / "artifacts/experiments/joint/robustness/missing-modalities/full"
            / f"seed-{seed}/{condition}.json"
        )
    return (
        root
        / "artifacts/autodl/robustness-results-v1/download-v1"
        / "bimer-robustness-results-v1/reports/robustness-evaluation"
        / condition
        / f"seed-{seed}.json"
    )


def test_metrics(payload: dict[str, Any]) -> dict[str, Any]:
    test = payload.get("test")
    if not isinstance(test, dict):
        raise ValueError("result does not contain a test metrics object")
    return test


def load_condition_runs(root: Path, condition: str) -> dict[int, dict[str, Any]]:
    runs: dict[int, dict[str, Any]] = {}
    for seed in SEEDS:
        path = result_path(root, condition, seed)
        if not path.exists():
            raise FileNotFoundError(path)
        runs[seed] = test_metrics(load_json(path))
    return runs


def summarize_metric(values: list[float]) -> tuple[float, float]:
    return mean(values), sample_stdev(values)


def summarize_condition(
    condition: str,
    runs: dict[int, dict[str, Any]],
    standard_runs: dict[int, dict[str, Any]],
) -> list[dict[str, Any]]:
    group, label = CONDITION_META[condition]
    rows: list[dict[str, Any]] = []

    for dataset in (*DATASETS, "bilingual_average"):
        per_seed: dict[str, list[float]] = {
            "weighted_f1": [],
            "macro_f1": [],
            "accuracy": [],
        }
        standard_weighted: list[float] = []

        for seed in SEEDS:
            if dataset in DATASETS:
                current_metrics = runs[seed][dataset]
                standard_metrics = standard_runs[seed][dataset]
                for metric in per_seed:
                    per_seed[metric].append(float(current_metrics[metric]))
                standard_weighted.append(float(standard_metrics["weighted_f1"]))
            else:
                for metric in per_seed:
                    per_seed[metric].append(
                        mean(float(runs[seed][name][metric]) for name in DATASETS)
                    )
                standard_weighted.append(
                    mean(float(standard_runs[seed][name]["weighted_f1"]) for name in DATASETS)
                )

        weighted_mean, weighted_std = summarize_metric(per_seed["weighted_f1"])
        macro_mean, macro_std = summarize_metric(per_seed["macro_f1"])
        accuracy_mean, accuracy_std = summarize_metric(per_seed["accuracy"])
        standard_mean = mean(standard_weighted)
        delta = weighted_mean - standard_mean

        rows.append(
            {
                "group": group,
                "condition": condition,
                "condition_zh": label,
                "dataset": dataset,
                "runs": len(SEEDS),
                "weighted_f1_mean": weighted_mean,
                "weighted_f1_std": weighted_std,
                "macro_f1_mean": macro_mean,
                "macro_f1_std": macro_std,
                "accuracy_mean": accuracy_mean,
                "accuracy_std": accuracy_std,
                "delta_from_standard": delta,
                "relative_delta_pct": (delta / standard_mean * 100.0)
                if not math.isclose(standard_mean, 0.0)
                else None,
                "per_seed_weighted_f1": {
                    str(seed): per_seed["weighted_f1"][index] for index, seed in enumerate(SEEDS)
                },
            }
        )
    return rows


def summarize_per_class(
    condition: str,
    runs: dict[int, dict[str, Any]],
    standard_runs: dict[int, dict[str, Any]],
) -> list[dict[str, Any]]:
    group, label = CONDITION_META[condition]
    rows: list[dict[str, Any]] = []
    for dataset in DATASETS:
        for emotion in CLASSES:
            values = [float(runs[seed][dataset]["per_class_f1"][emotion]) for seed in SEEDS]
            standard_values = [
                float(standard_runs[seed][dataset]["per_class_f1"][emotion]) for seed in SEEDS
            ]
            rows.append(
                {
                    "group": group,
                    "condition": condition,
                    "condition_zh": label,
                    "dataset": dataset,
                    "emotion": emotion,
                    "f1_mean": mean(values),
                    "f1_std": sample_stdev(values),
                    "delta_from_standard": mean(values) - mean(standard_values),
                }
            )
    return rows


def sample_id(row: dict[str, Any]) -> str:
    return ":".join(
        (
            str(row["dataset"]),
            str(row["split"]),
            str(row["dialogue_id"]),
            str(row["utterance_id"]),
        )
    )


def normalize_words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.lower())


def normalize_characters(text: str) -> list[str]:
    return re.findall(r"[\u4e00-\u9fffA-Za-z0-9]", text.lower())


def edit_distance(reference: list[str], hypothesis: list[str]) -> int:
    if len(reference) < len(hypothesis):
        reference, hypothesis = hypothesis, reference
    previous = list(range(len(hypothesis) + 1))
    for ref_index, ref_item in enumerate(reference, start=1):
        current = [ref_index]
        for hyp_index, hyp_item in enumerate(hypothesis, start=1):
            substitution = previous[hyp_index - 1] + (ref_item != hyp_item)
            insertion = current[hyp_index - 1] + 1
            deletion = previous[hyp_index] + 1
            current.append(min(substitution, insertion, deletion))
        previous = current
    return previous[-1]


def summarize_whisper(root: Path) -> list[dict[str, Any]]:
    package_root = (
        root / "artifacts/autodl/robustness-results-v1/download-v1" / "bimer-robustness-results-v1"
    )
    whisper_rows = load_jsonl(package_root / "output/whisper-test.jsonl")
    errors = load_jsonl(package_root / "reports/whisper-test-errors.jsonl")
    error_ids = {str(row["sample_id"]) for row in errors}

    original_rows = {
        sample_id(row): row
        for row in load_jsonl(root / "data/processed/all.jsonl")
        if row.get("split") == "test"
    }

    summaries: list[dict[str, Any]] = []
    for dataset in DATASETS:
        dataset_rows = [row for row in whisper_rows if row.get("dataset") == dataset]
        success_rows = [row for row in dataset_rows if sample_id(row) not in error_ids]
        edit_total = 0
        reference_units = 0
        exact_matches = 0
        modified_inputs = 0

        for row in dataset_rows:
            key = sample_id(row)
            reference = str(original_rows[key]["text"])
            hypothesis = str(row["text"])
            normalizer = normalize_words if dataset == "meld" else normalize_characters
            reference_tokens = normalizer(reference)
            hypothesis_tokens = normalizer(hypothesis)
            if reference_tokens != hypothesis_tokens:
                modified_inputs += 1
            if key in error_ids:
                continue
            if reference_tokens == hypothesis_tokens:
                exact_matches += 1
            edit_total += edit_distance(reference_tokens, hypothesis_tokens)
            reference_units += len(reference_tokens)

        fallback_count = len(dataset_rows) - len(success_rows)
        summaries.append(
            {
                "dataset": dataset,
                "metric": "WER" if dataset == "meld" else "CER",
                "samples": len(dataset_rows),
                "asr_successes": len(success_rows),
                "fallback_to_original": fallback_count,
                "fallback_rate": fallback_count / len(dataset_rows),
                "exact_match_rate_on_success": exact_matches / len(success_rows),
                "corpus_edit_error_rate_on_success": edit_total / reference_units,
                "modified_pipeline_inputs": modified_inputs,
                "modified_pipeline_input_rate": modified_inputs / len(dataset_rows),
            }
        )
    return summaries


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    root = args.project_root.resolve()
    output_dir = (
        args.output_dir.resolve() if args.output_dir else root / "artifacts/analysis/robustness"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    standard_runs = load_condition_runs(root, "standard")
    all_runs = {condition: load_condition_runs(root, condition) for condition in CONDITION_META}

    summary_rows: list[dict[str, Any]] = []
    per_class_rows: list[dict[str, Any]] = []
    for condition in CONDITION_META:
        runs = all_runs[condition]
        summary_rows.extend(summarize_condition(condition, runs, standard_runs))
        per_class_rows.extend(summarize_per_class(condition, runs, standard_runs))

    whisper_rows = summarize_whisper(root)

    write_csv(
        output_dir / "robustness-summary.csv",
        summary_rows,
        [
            "group",
            "condition",
            "condition_zh",
            "dataset",
            "runs",
            "weighted_f1_mean",
            "weighted_f1_std",
            "macro_f1_mean",
            "macro_f1_std",
            "accuracy_mean",
            "accuracy_std",
            "delta_from_standard",
            "relative_delta_pct",
        ],
    )
    write_csv(
        output_dir / "robustness-per-class.csv",
        per_class_rows,
        [
            "group",
            "condition",
            "condition_zh",
            "dataset",
            "emotion",
            "f1_mean",
            "f1_std",
            "delta_from_standard",
        ],
    )
    write_csv(
        output_dir / "whisper-transcription-quality.csv",
        whisper_rows,
        list(whisper_rows[0].keys()),
    )

    payload = {
        "metadata": {
            "seeds": list(SEEDS),
            "datasets": list(DATASETS),
            "condition_order": list(CONDITION_META),
            "inference_note": (
                "No confidence interval is synthesized from independent seed-level "
                "endpoints. V2 model comparisons use paired cluster bootstrap over "
                "complete dialogue IDs via scripts/compare_v2_predictions.py."
            ),
        },
        "conditions": summary_rows,
        "per_class": per_class_rows,
        "whisper_transcription_quality": whisper_rows,
    }
    with (output_dir / "robustness-summary.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    print(f"wrote {len(summary_rows)} summary rows")
    print(f"wrote {len(per_class_rows)} per-class rows")
    print(f"output: {output_dir}")


if __name__ == "__main__":
    main()

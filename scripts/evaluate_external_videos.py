#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from bimer.external_evaluation import (
    EXTERNAL_CONDITIONS,
    annotation_agreement,
    evaluate_external_predictions,
    external_model_acceptance,
)
from bimer.labels import EMOTION_LABELS


def _rows(path: Path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _by_key(rows):
    values = {(row["video_id"], row["segment_id"]): row for row in rows}
    if len(values) != len(rows):
        raise ValueError("duplicate video_id/segment_id rows")
    return values


def _evaluate(plan, adjudicated, predictions, iterations):
    plan_by_id = {video["video_id"]: video for video in plan["videos"]}
    truth_by_key = _by_key(adjudicated)
    prediction_by_key = _by_key(predictions)
    if set(truth_by_key) != set(prediction_by_key):
        raise ValueError("adjudicated labels and predictions do not align")
    ordered = sorted(truth_by_key)
    truth = np.asarray(
        [EMOTION_LABELS.index(truth_by_key[key]["label"]) for key in ordered],
        dtype=np.int64,
    )
    probabilities = np.asarray(
        [
            [float(prediction_by_key[key][f"probability_{label}"]) for label in EMOTION_LABELS]
            for key in ordered
        ],
        dtype=np.float64,
    )
    video_ids = np.asarray([key[0] for key in ordered])
    conditions = np.asarray(
        [plan_by_id[key[0]]["condition"] for key in ordered],
        dtype=str,
    )
    report = evaluate_external_predictions(
        truth,
        probabilities,
        video_ids=video_ids,
        conditions=conditions,
        label_names=EMOTION_LABELS,
        bootstrap_iterations=iterations,
        seed=42,
    )
    confidence_status = np.asarray(
        [prediction_by_key[key].get("confidence_status", "confident") for key in ordered]
    )
    runtime = np.asarray(
        [float(prediction_by_key[key].get("runtime_seconds", 0.0)) for key in ordered]
    )
    report["confidence_coverage"] = float((confidence_status == "confident").mean())
    report["runtime_seconds"] = {
        "mean": float(runtime.mean()),
        "p95": float(np.quantile(runtime, 0.95)),
    }
    for condition in EXTERNAL_CONDITIONS:
        active = conditions == condition
        if condition in report["by_condition"]:
            report["by_condition"][condition]["confidence_coverage"] = float(
                (confidence_status[active] == "confident").mean()
            )
            report["by_condition"][condition]["runtime_seconds_mean"] = float(
                runtime[active].mean()
            )
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--annotator-one", required=True, type=Path)
    parser.add_argument("--annotator-two", required=True, type=Path)
    parser.add_argument("--adjudicated", required=True, type=Path)
    parser.add_argument("--v2-predictions", required=True, type=Path)
    parser.add_argument("--comparison-predictions", type=Path)
    parser.add_argument("--comparison-name", default="candidate")
    parser.add_argument(
        "--v3-predictions",
        type=Path,
        help="Deprecated alias for --comparison-predictions.",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--iterations", type=int, default=2000)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    first = _by_key(_rows(args.annotator_one))
    second = _by_key(_rows(args.annotator_two))
    if set(first) != set(second):
        raise SystemExit("annotator files do not contain the same segments")
    ordered = sorted(first)
    agreement = annotation_agreement(
        [first[key]["label"] for key in ordered],
        [second[key]["label"] for key in ordered],
    )
    if agreement["requires_reannotation"]:
        raise SystemExit("Cohen's kappa is below 0.60; reannotation is required")
    adjudicated = _rows(args.adjudicated)
    v2 = _evaluate(plan, adjudicated, _rows(args.v2_predictions), args.iterations)
    payload = {
        "annotation_agreement": agreement,
        "v2": v2,
        "conditions": list(EXTERNAL_CONDITIONS),
    }
    comparison_path = args.comparison_predictions or args.v3_predictions
    if comparison_path is not None:
        comparison_name = "v3" if args.v3_predictions is not None else args.comparison_name
        comparison = _evaluate(
            plan,
            adjudicated,
            _rows(comparison_path),
            args.iterations,
        )
        payload["comparison_name"] = comparison_name
        payload["comparison"] = comparison
        payload["comparison_acceptance"] = external_model_acceptance(v2, comparison)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

from .metrics import paired_cluster_bootstrap_weighted_f1_delta


def compare_prediction_archives(
    baseline_path: Path | str,
    candidate_path: Path | str,
    output_path: Path | str,
    *,
    iterations: int = 2000,
    seed: int = 42,
) -> Path:
    with np.load(baseline_path, allow_pickle=False) as payload:
        baseline = {name: payload[name] for name in payload.files}
    with np.load(candidate_path, allow_pickle=False) as payload:
        candidate = {name: payload[name] for name in payload.files}
    baseline_ids = baseline["sample_ids"].astype(str)
    candidate_ids = candidate["sample_ids"].astype(str)
    if len(set(baseline_ids)) != len(baseline_ids) or len(set(candidate_ids)) != len(
        candidate_ids
    ):
        raise ValueError("prediction archives contain duplicate sample IDs")
    if set(baseline_ids) != set(candidate_ids):
        raise ValueError("prediction archives contain different sample IDs")
    candidate_position = {
        sample_id: index for index, sample_id in enumerate(candidate_ids)
    }
    order = np.asarray([candidate_position[sample_id] for sample_id in baseline_ids])
    candidate_truth = candidate["truth"][order]
    candidate_contexts = candidate["context_ids"].astype(str)[order]
    truth = baseline["truth"]
    contexts = baseline["context_ids"].astype(str)
    if not np.array_equal(truth, candidate_truth):
        raise ValueError("prediction archives disagree on ground truth")
    if not np.array_equal(contexts, candidate_contexts):
        raise ValueError("prediction archives disagree on context IDs")
    baseline_prediction = baseline["prediction"]
    candidate_prediction = candidate["prediction"][order]
    baseline_score = float(
        f1_score(truth, baseline_prediction, average="weighted", zero_division=0)
    )
    candidate_score = float(
        f1_score(truth, candidate_prediction, average="weighted", zero_division=0)
    )
    interval = paired_cluster_bootstrap_weighted_f1_delta(
        truth,
        baseline_prediction,
        candidate_prediction,
        contexts,
        iterations=iterations,
        seed=seed,
    )
    payload = {
        "baseline": str(baseline_path),
        "candidate": str(candidate_path),
        "samples": len(truth),
        "contexts": len(np.unique(contexts)),
        "baseline_weighted_f1": baseline_score,
        "candidate_weighted_f1": candidate_score,
        "weighted_f1_delta": candidate_score - baseline_score,
        "weighted_f1_delta_ci95": list(interval),
        "bootstrap_unit": "context",
        "paired": True,
        "significant_at_0_05": bool(interval[0] > 0 or interval[1] < 0),
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(output)
    return output

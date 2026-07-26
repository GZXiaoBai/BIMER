import json

import numpy as np

from bimer.prediction_comparison import compare_prediction_archives


def test_prediction_comparison_reorders_and_bootstraps_paired_contexts(tmp_path):
    baseline = tmp_path / "baseline.npz"
    candidate = tmp_path / "candidate.npz"
    sample_ids = np.asarray(["a", "b", "c", "d"])
    contexts = np.asarray(["x", "x", "y", "y"])
    truth = np.asarray([0, 1, 0, 1])
    np.savez_compressed(
        baseline,
        sample_ids=sample_ids,
        context_ids=contexts,
        truth=truth,
        prediction=np.asarray([0, 0, 0, 0]),
    )
    order = np.asarray([2, 0, 3, 1])
    np.savez_compressed(
        candidate,
        sample_ids=sample_ids[order],
        context_ids=contexts[order],
        truth=truth[order],
        prediction=truth[order],
    )

    output = compare_prediction_archives(
        baseline, candidate, tmp_path / "comparison.json", iterations=100
    )
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert payload["bootstrap_unit"] == "context"
    assert payload["samples"] == 4
    assert payload["candidate_weighted_f1"] == 1.0
    assert payload["weighted_f1_delta"] > 0
    assert len(payload["weighted_f1_delta_ci95"]) == 2

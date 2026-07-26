import numpy as np
import pytest

from bimer.v4_analysis import (
    analyze_context_gates,
    context_stratified_metrics,
    ensemble_predictions,
    prototype_geometry,
    should_enable_ensemble,
)


def _prediction(path, probabilities, *, sample_ids=("a", "b", "c", "d")):
    probabilities = np.asarray(probabilities, dtype=np.float32)
    truth = np.asarray([0, 1, 1, 0], dtype=np.int64)
    np.savez_compressed(
        path,
        sample_ids=np.asarray(sample_ids),
        context_ids=np.asarray(["x", "x", "y", "y"]),
        truth=truth,
        prediction=probabilities.argmax(axis=1),
        probabilities=probabilities,
        gates=np.full((4, 3), 1 / 3, dtype=np.float32),
        context_lengths=np.asarray([4, 8, 12, 24]),
    )
    return path


def test_three_seed_ensemble_aligns_ids_and_normalizes_probabilities(tmp_path):
    paths = [
        _prediction(
            tmp_path / f"seed-{seed}.npz",
            [[0.9, 0.1], [0.2, 0.8], [0.3, 0.7], [0.8, 0.2]],
        )
        for seed in (42, 123, 2026)
    ]

    payload = ensemble_predictions(paths)

    assert payload["probabilities"].shape == (4, 2)
    assert np.allclose(payload["probabilities"].sum(axis=1), 1.0)
    assert payload["prediction"].tolist() == [0, 1, 1, 0]
    assert payload["sample_ids"].tolist() == ["a", "b", "c", "d"]


def test_ensemble_rejects_misaligned_sample_ids(tmp_path):
    first = _prediction(
        tmp_path / "first.npz",
        [[0.9, 0.1], [0.2, 0.8], [0.3, 0.7], [0.8, 0.2]],
    )
    second = _prediction(
        tmp_path / "second.npz",
        [[0.9, 0.1], [0.2, 0.8], [0.3, 0.7], [0.8, 0.2]],
        sample_ids=("b", "a", "c", "d"),
    )

    with pytest.raises(ValueError, match="align"):
        ensemble_predictions([first, second])


def test_ensemble_requires_weighted_gain_and_no_macro_regression():
    assert should_enable_ensemble(
        single={"weighted_f1": 0.60, "macro_f1": 0.45},
        ensemble={"weighted_f1": 0.603, "macro_f1": 0.45},
    )
    assert not should_enable_ensemble(
        single={"weighted_f1": 0.60, "macro_f1": 0.45},
        ensemble={"weighted_f1": 0.6029, "macro_f1": 0.46},
    )
    assert not should_enable_ensemble(
        single={"weighted_f1": 0.60, "macro_f1": 0.45},
        ensemble={"weighted_f1": 0.61, "macro_f1": 0.449},
    )


def test_context_strata_and_gate_conflict_analysis():
    truth = np.asarray([0, 1, 1, 0, 1, 0])
    prediction = np.asarray([0, 1, 0, 0, 1, 1])
    lengths = np.asarray([1, 8, 9, 16, 17, 32])
    gates = np.asarray([0.1, 0.2, 0.4, 0.5, 0.8, 0.9])
    local = np.asarray([0, 1, 0, 0, 0, 0])
    fixed = np.asarray([0, 1, 1, 0, 1, 1])

    strata = context_stratified_metrics(truth, prediction, lengths, label_names=("n", "j"))
    relationship = analyze_context_gates(gates, lengths, local, fixed)

    assert set(strata) == {"1-8", "9-16", "17-32"}
    assert all(values["samples"] == 2 for values in strata.values())
    assert relationship["mean_gate_by_length"]["1-8"] == pytest.approx(0.15)
    assert relationship["conflict"]["samples"] == 3
    assert relationship["conflict"]["mean_gate"] > relationship["agreement"]["mean_gate"]


def test_prototype_geometry_reports_lower_within_than_between_distance():
    representations = np.asarray(
        [[1.0, 0.0], [0.9, 0.1], [-1.0, 0.0], [-0.9, -0.1]],
        dtype=np.float32,
    )
    truth = np.asarray([0, 0, 1, 1])

    geometry = prototype_geometry(representations, truth, num_classes=2)

    assert geometry["within_class_distance"] < geometry["between_class_distance"]
    assert geometry["separation"] > 0

import importlib.util
from pathlib import Path

import pytest


def _module():
    path = Path(__file__).parents[1] / "scripts" / "summarize_robustness_results.py"
    spec = importlib.util.spec_from_file_location("robustness_summary", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_robustness_summary_uses_sample_standard_deviation():
    mean, standard_deviation = _module().summarize_metric([0.4, 0.6])

    assert mean == pytest.approx(0.5)
    assert standard_deviation == pytest.approx(0.1414213562)


def test_robustness_summary_does_not_average_independent_ci_endpoints():
    module = _module()
    metrics = {
        dataset: {
            "weighted_f1": 0.5,
            "macro_f1": 0.4,
            "accuracy": 0.6,
            "weighted_f1_ci95": [0.1, 0.9],
        }
        for dataset in module.DATASETS
    }
    runs = {seed: metrics for seed in module.SEEDS}

    rows = module.summarize_condition("standard", runs, runs)

    assert all("weighted_f1_ci95_low_mean" not in row for row in rows)
    assert all("weighted_f1_ci95_high_mean" not in row for row in rows)

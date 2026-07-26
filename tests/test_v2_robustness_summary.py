from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "summarize_v2_robustness_results.py"


def _module():
    spec = importlib.util.spec_from_file_location("v2_robustness_summary", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _fixture(root: Path) -> tuple[Path, Path]:
    module = _module()
    results = root / "results"
    selection = root / "selection.json"
    truth = np.asarray([0, 1, 0, 1], dtype=np.int64)
    sample_ids = np.asarray(["a", "b", "c", "d"])
    context_ids = np.asarray(
        ["dialogue-1", "dialogue-1", "dialogue-2", "dialogue-2"]
    )
    for model in module.MODELS:
        for condition in module.CONDITION_META:
            for seed in module.SEEDS:
                prediction = (
                    truth.copy()
                    if model == "quality_lagf"
                    else np.zeros_like(truth)
                )
                weighted_f1 = 1.0 if model == "quality_lagf" else 1 / 3
                output = results / model / condition / f"seed-{seed}.json"
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_text(
                    json.dumps(
                        {
                            "test": {
                                dataset: {
                                    "weighted_f1": weighted_f1,
                                    "macro_f1": weighted_f1,
                                    "accuracy": float(
                                        np.mean(prediction == truth)
                                    ),
                                }
                                for dataset in module.DATASETS
                            }
                        }
                    ),
                    encoding="utf-8",
                )
                prediction_dir = (
                    results
                    / model
                    / condition
                    / f"seed-{seed}.predictions"
                )
                prediction_dir.mkdir(parents=True, exist_ok=True)
                for dataset in module.DATASETS:
                    np.savez_compressed(
                        prediction_dir / f"{dataset}.npz",
                        sample_ids=sample_ids,
                        context_ids=context_ids,
                        truth=truth,
                        prediction=prediction,
                    )
    selection.write_text(
        json.dumps(
            {
                "status": "frozen",
                "selection_scope": "validation_only",
                "test_set_used_for_selection": False,
                "selected_model": "quality_lagf",
                "validation_weighted_f1": {
                    "bilingual_average": 0.61,
                },
                "no_gate_comparison": {
                    "bilingual_average": 0.60,
                    "selected_model_delta": 0.01,
                },
            }
        ),
        encoding="utf-8",
    )
    return results, selection


def test_sample_standard_deviation_uses_ddof_one():
    module = _module()

    assert module.sample_stdev([0.4, 0.6]) == pytest.approx(0.1414213562)


def test_paired_bootstrap_resamples_complete_contexts_deterministically():
    module = _module()
    truth = np.asarray([0, 1, 0, 1], dtype=np.int64)
    bundle = module.PredictionBundle(
        sample_ids=np.asarray(["a", "b", "c", "d"]),
        context_ids=np.asarray(["x", "x", "y", "y"]),
        truth=truth,
        prediction=truth,
    )
    baseline = module.PredictionBundle(
        sample_ids=bundle.sample_ids,
        context_ids=bundle.context_ids,
        truth=truth,
        prediction=np.zeros_like(truth),
    )

    first = module.paired_cluster_delta_draws(
        bundle,
        baseline,
        iterations=100,
        rng=np.random.default_rng(42),
    )
    second = module.paired_cluster_delta_draws(
        bundle,
        baseline,
        iterations=100,
        rng=np.random.default_rng(42),
    )

    assert first[0] == pytest.approx(2 / 3)
    assert first[2] == 2
    assert np.array_equal(first[1], second[1])
    assert np.all(first[1] >= 0)


def test_cli_validates_full_matrix_and_writes_decision(tmp_path):
    results, selection = _fixture(tmp_path)
    output = tmp_path / "analysis"

    run = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--results-root",
            str(results),
            "--selection-config",
            str(selection),
            "--output-dir",
            str(output),
            "--bootstrap-iterations",
            "100",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )

    assert run.returncode == 0, run.stderr
    validation = json.loads((output / "validation.json").read_text())
    decision = json.loads((output / "selection-decision.json").read_text())
    assert validation["result_json_found"] == 72
    assert validation["prediction_npz_found"] == 144
    assert validation["summary_rows"] == 72
    assert validation["comparison_rows"] == 36
    assert validation["json_prediction_weighted_f1_match"] is True
    assert decision["decision"] == "deploy_quality_lagf"

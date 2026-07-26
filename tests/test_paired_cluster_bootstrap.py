import csv
from pathlib import Path
import subprocess
import sys

import numpy as np

from test_v2_formal_summary import _build_complete_fixture


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "paired_cluster_bootstrap_v2.py"


def _add_predictions(results_root: Path) -> None:
    truth = np.asarray([0, 1, 0, 1], dtype=np.int64)
    sample_ids = np.asarray(["a", "b", "c", "d"])
    context_ids = np.asarray(["dialogue-1", "dialogue-1", "dialogue-2", "dialogue-2"])
    for result_path in results_root.glob("**/results.json"):
        prediction = (
            truth.copy()
            if "/formal/quality_lagf/" in str(result_path)
            else np.zeros_like(truth)
        )
        prediction_dir = result_path.parent / "predictions"
        prediction_dir.mkdir(parents=True, exist_ok=True)
        for dataset in ("meld", "emotiontalk"):
            np.savez_compressed(
                prediction_dir / f"{dataset}.npz",
                sample_ids=sample_ids,
                context_ids=context_ids,
                truth=truth,
                prediction=prediction,
            )


def test_cli_writes_paired_dialogue_cluster_confidence_intervals(tmp_path):
    results = tmp_path / "results"
    output = tmp_path / "paired_cluster_bootstrap.csv"
    _build_complete_fixture(results)
    _add_predictions(results)

    run = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--input",
            str(results),
            "--output",
            str(output),
            "--iterations",
            "200",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )

    assert run.returncode == 0, run.stderr
    with output.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 27
    row = next(
        item
        for item in rows
        if item["comparison_scope"] == "formal"
        and item["comparator"] == "early_mlp"
        and item["dataset"] == "bilingual_average"
    )
    assert np.isclose(
        float(row["weighted_f1_delta_full_minus_comparator"]),
        2 / 3,
    )
    assert float(row["ci95_lower"]) > 0
    assert float(row["ci95_upper"]) > 0
    assert row["supports_full"] == "True"


def test_cli_rejects_misaligned_prediction_ids(tmp_path):
    results = tmp_path / "results"
    output = tmp_path / "paired_cluster_bootstrap.csv"
    _build_complete_fixture(results)
    _add_predictions(results)
    target = next(
        (results / "formal" / "early_mlp").glob(
            "**/seed-42/predictions/meld.npz"
        )
    )
    with np.load(target) as payload:
        arrays = {key: payload[key] for key in payload.files}
    arrays["sample_ids"] = np.asarray(["a", "b", "c", "wrong"])
    np.savez_compressed(target, **arrays)

    run = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--input",
            str(results),
            "--output",
            str(output),
            "--iterations",
            "20",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )

    assert run.returncode != 0
    assert "sample ids" in run.stderr.lower()

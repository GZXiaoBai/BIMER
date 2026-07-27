import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def test_calibration_script_writes_profile_report_and_reliability_figure(tmp_path):
    predictions = tmp_path / "predictions"
    predictions.mkdir()
    for dataset, language in (("meld", "en"), ("emotiontalk", "zh")):
        np.savez_compressed(
            predictions / f"{dataset}.npz",
            probabilities=np.asarray(
                [[0.9, 0.1], [0.8, 0.2], [0.2, 0.8], [0.1, 0.9]],
                dtype=np.float32,
            ),
            truth=np.asarray([0, 1, 1, 0], dtype=np.int64),
            languages=np.asarray([language] * 4),
        )
    profile = tmp_path / "calibration.json"
    report = tmp_path / "report.json"
    figure = tmp_path / "reliability.png"
    environment = {
        **os.environ,
        "PYTHONPATH": str(ROOT / "src"),
    }

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "fit_v3_calibration.py"),
            "--predictions",
            str(predictions),
            "--profile",
            str(profile),
            "--report",
            str(report),
            "--figure",
            str(figure),
        ],
        cwd=ROOT,
        env=environment,
        check=True,
    )

    assert set(json.loads(profile.read_text())["languages"]) == {"en", "zh"}
    assert json.loads(report.read_text())["test_set_used"] is False
    assert figure.read_bytes().startswith(b"\x89PNG")


def test_calibration_script_infers_language_for_standard_dataset_predictions(tmp_path):
    predictions = tmp_path / "predictions"
    predictions.mkdir()
    for dataset in ("meld", "emotiontalk"):
        np.savez_compressed(
            predictions / f"{dataset}.npz",
            probabilities=np.asarray(
                [[0.9, 0.1], [0.8, 0.2], [0.2, 0.8], [0.1, 0.9]],
                dtype=np.float32,
            ),
            truth=np.asarray([0, 1, 1, 0], dtype=np.int64),
        )
    profile = tmp_path / "calibration.json"
    report = tmp_path / "report.json"
    figure = tmp_path / "reliability.png"

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "fit_v3_calibration.py"),
            "--predictions",
            str(predictions),
            "--profile",
            str(profile),
            "--report",
            str(report),
            "--figure",
            str(figure),
        ],
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
        check=True,
    )

    assert set(json.loads(profile.read_text())["languages"]) == {"en", "zh"}

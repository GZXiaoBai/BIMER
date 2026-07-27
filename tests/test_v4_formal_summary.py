import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "summarize_v4_formal.py"
VARIANTS = ("full", "no_context_gate", "no_prototype", "neither")
SEEDS = (42, 123, 2026)


def _write_run(root: Path, variant: str, seed: int) -> None:
    run = root / variant / "adaptive_context_prototype" / "joint" / f"seed-{seed}"
    predictions = run / "validation_predictions"
    predictions.mkdir(parents=True)
    validation = {}
    for dataset in ("meld", "emotiontalk"):
        truth = np.asarray([0, 1, 1, 0], dtype=np.int64)
        gain = 0.02 if variant == "full" else 0.0
        weighted = 0.60 + gain + (seed % 3) * 0.001
        validation[dataset] = {
            "weighted_f1": weighted,
            "macro_f1": 0.45 + gain,
            "accuracy": 0.62 + gain,
            "per_class_f1": {
                "neutral": 0.7,
                "joy": 0.6,
                "sadness": 0.4,
                "anger": 0.5,
                "surprise": 0.4,
                "fear": 0.2,
                "disgust": 0.1,
            },
        }
        probabilities = np.asarray(
            [[0.8, 0.2], [0.1, 0.9], [0.3, 0.7], [0.7, 0.3]],
            dtype=np.float32,
        )
        np.savez_compressed(
            predictions / f"{dataset}.npz",
            sample_ids=np.asarray([f"{dataset}-{index}" for index in range(4)]),
            context_ids=np.asarray([f"{dataset}-a"] * 2 + [f"{dataset}-b"] * 2),
            truth=truth,
            prediction=probabilities.argmax(axis=1),
            probabilities=probabilities,
            gates=np.full((4, 3), 1 / 3, dtype=np.float32),
            context_lengths=np.asarray([4, 8, 12, 24]),
            context_gates=np.asarray([0.2, 0.3, 0.6, 0.8], dtype=np.float32),
            local_prediction=np.asarray([0, 1, 0, 0]),
            fixed_context_prediction=np.asarray([0, 1, 1, 1]),
            representations=np.asarray(
                [[1.0, 0.0], [-1.0, 0.0], [-0.9, 0.1], [0.9, -0.1]],
                dtype=np.float32,
            ),
        )
    (run / "results.json").write_text(
        json.dumps({"validation": validation, "test": {}}),
        encoding="utf-8",
    )


def test_formal_summary_reports_ddof1_ensemble_and_v4_evidence(tmp_path):
    formal = tmp_path / "formal"
    for variant in VARIANTS:
        for seed in SEEDS:
            _write_run(formal, variant, seed)
    output = tmp_path / "formal-summary.json"

    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--formal-root",
            str(formal),
            "--output",
            str(output),
            "--bootstrap-iterations",
            "20",
        ],
        cwd=ROOT,
        check=True,
    )

    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["seeds"] == [42, 123, 2026]
    assert report["variants"]["full"]["meld"]["weighted_f1"]["std"] > 0
    assert report["ensemble"]["probability_average"] is True
    assert set(report["evidence"]["context_strata"]["meld"]) == {"1-8", "9-16", "17-32"}
    assert "prototype_geometry" in report["evidence"]
    assert set(report["paired_bootstrap"]) == {
        "no_context_gate",
        "no_prototype",
        "neither",
    }

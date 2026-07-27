from __future__ import annotations

import json

import pytest

from scripts.summarize_v2_unimodal import summarize_unimodal_results


def test_unimodal_summary_reports_three_seed_sample_statistics(tmp_path) -> None:
    root = tmp_path / "formal"
    for seed, weighted_f1 in ((42, 0.40), (123, 0.50), (2026, 0.60)):
        directory = root / "audio" / "audio" / "meld" / f"seed-{seed}"
        directory.mkdir(parents=True)
        (directory / "results.json").write_text(
            json.dumps(
                {
                    "config": {"learning_rate": 0.001},
                    "test": {
                        "meld": {
                            "weighted_f1": weighted_f1,
                            "macro_f1": weighted_f1 - 0.1,
                            "accuracy": weighted_f1 + 0.05,
                            "confusion_matrix": [[2, 0], [0, 2]],
                        }
                    },
                }
            ),
            encoding="utf-8",
        )

    rows = summarize_unimodal_results(
        root,
        models=("audio",),
        datasets=("meld",),
    )

    assert len(rows) == 1
    assert rows[0]["weighted_f1_mean"] == pytest.approx(0.5)
    assert rows[0]["weighted_f1_std"] == pytest.approx(0.1)
    assert rows[0]["predicted_class_count_min"] == 2
    assert rows[0]["learning_rate"] == 0.001

from __future__ import annotations

import pandas as pd

from scripts.generate_public_results import (
    select_public_per_class,
    select_public_robustness,
    v3_public_summary,
)


def test_v3_public_summary_records_protocol_stop() -> None:
    summary = v3_public_summary(
        {
            "selected": "weighted_ce",
            "passed": [],
            "diagnostics": {"focal": {"accepted": False}},
        },
        {
            "selected": 0.0,
            "passed": [],
            "diagnostics": {"0.1": {"accepted": False}},
        },
    )

    assert summary["decision"] == "stop_v3_and_deploy_v2"
    assert summary["formal_v3_test_run"] is False
    assert summary["loss_screen"]["selected"] == "weighted_ce"
    assert summary["ranking_screen"]["selected_lambda"] == 0.0


def test_public_robustness_keeps_only_aggregate_supported_scope() -> None:
    frame = pd.DataFrame(
        [
            {
                "model": "quality_lagf",
                "condition": "standard",
                "dataset": "bilingual_average",
                "runs": 3,
                "weighted_f1_mean": 0.60,
                "weighted_f1_std": 0.01,
                "macro_f1_mean": 0.47,
                "macro_f1_std": 0.01,
                "accuracy_mean": 0.60,
                "accuracy_std": 0.01,
                "delta_from_standard": 0.0,
            },
            {
                "model": "quality_lagf",
                "condition": "standard",
                "dataset": "meld",
                "runs": 3,
                "weighted_f1_mean": 0.58,
                "weighted_f1_std": 0.01,
                "macro_f1_mean": 0.39,
                "macro_f1_std": 0.01,
                "accuracy_mean": 0.59,
                "accuracy_std": 0.01,
                "delta_from_standard": 0.0,
            },
            {
                "model": "other",
                "condition": "standard",
                "dataset": "bilingual_average",
                "runs": 3,
                "weighted_f1_mean": 0.50,
                "weighted_f1_std": 0.01,
                "macro_f1_mean": 0.40,
                "macro_f1_std": 0.01,
                "accuracy_mean": 0.50,
                "accuracy_std": 0.01,
                "delta_from_standard": 0.0,
            },
        ]
    )

    selected = select_public_robustness(frame)

    assert selected[["model", "condition"]].to_dict("records") == [
        {"model": "quality_lagf", "condition": "standard"}
    ]
    assert "dataset" not in selected.columns


def test_public_per_class_excludes_ablations() -> None:
    frame = pd.DataFrame(
        [
            {
                "scope": "formal",
                "variant": "quality_lagf",
                "dataset": "meld",
                "label": "joy",
                "seed_count": 3,
                "f1_mean": 0.52,
                "f1_std": 0.03,
            },
            {
                "scope": "ablations",
                "variant": "no_context",
                "dataset": "meld",
                "label": "joy",
                "seed_count": 3,
                "f1_mean": 0.53,
                "f1_std": 0.01,
            },
        ]
    )

    selected = select_public_per_class(frame)

    assert selected["variant"].tolist() == ["quality_lagf"]

from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_v5_public_summary_matches_frozen_stop_decision() -> None:
    with (ROOT / "results/v5_validation_summary.csv").open(
        encoding="utf-8",
        newline="",
    ) as handle:
        rows = {row["variant"]: row for row in csv.DictReader(handle)}

    assert set(rows) == {"v2_baseline", "beta_005", "beta_010"}
    assert rows["beta_005"]["decision"] == "failed"
    assert rows["beta_010"]["decision"] == "failed"
    assert float(rows["beta_005"]["whisper_weighted_f1_gain"]) < 0.015
    assert float(rows["beta_010"]["whisper_weighted_f1_gain"]) < 0.015
    assert float(rows["beta_005"]["emotiontalk_whisper_gain"]) < 0.0
    assert float(rows["beta_010"]["emotiontalk_whisper_gain"]) < 0.0
    assert float(rows["beta_005"]["video_drop_50_weighted_f1_delta"]) < -0.005
    assert float(rows["beta_010"]["video_drop_50_weighted_f1_delta"]) < -0.005


def test_v5_public_narrative_preserves_test_boundary_and_archive_hash() -> None:
    report = (ROOT / "docs/v5_exploratory_results.md").read_text(encoding="utf-8")
    model_card = " ".join((ROOT / "MODEL_CARD.md").read_text(encoding="utf-8").split())
    digest = "22e6750e900209ff5917ca697676e716e81afd2cc67a2e696596a93497de7cb1"

    assert "`stop_v5`" in report
    assert "不访问 MELD 或 EmotionTalk 官方测试集" in report
    assert digest in report
    assert "never accessed the official test sets" in model_card
    assert "continues to deploy V2" in model_card

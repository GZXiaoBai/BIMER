from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.freeze_v5_selection import freeze_from_decision
from scripts.summarize_v5_screen import summarize_v5_screen


def _write_result(
    root: Path,
    *,
    clean: tuple[float, float],
    whisper: tuple[float, float],
    audio: tuple[float, float],
    video: tuple[float, float],
) -> Path:
    result = root / "results.json"
    result.parent.mkdir(parents=True, exist_ok=True)

    def metrics(values):
        return {
            dataset: {"weighted_f1": values[0], "macro_f1": values[1]}
            for dataset in ("meld", "emotiontalk")
        }

    result.write_text(json.dumps({"validation": metrics(clean)}), encoding="utf-8")
    conditions = result.parent / "validation_conditions"
    conditions.mkdir()
    for name, values in (
        ("whisper", whisper),
        ("audio_10db", audio),
        ("video_drop_50", video),
    ):
        (conditions / f"{name}.json").write_text(
            json.dumps({"validation": metrics(values)}),
            encoding="utf-8",
        )
    return result


def test_v5_summary_freezes_only_validation_evidence(tmp_path: Path) -> None:
    baseline = _write_result(
        tmp_path / "baseline",
        clean=(0.60, 0.45),
        whisper=(0.55, 0.40),
        audio=(0.57, 0.42),
        video=(0.56, 0.41),
    )
    (baseline.parent / "validation_conditions" / "video_drop_50.json").rename(
        baseline.parent / "validation_conditions" / "video_50.json"
    )
    candidate = _write_result(
        tmp_path / "candidate",
        clean=(0.598, 0.448),
        whisper=(0.57, 0.42),
        audio=(0.566, 0.42),
        video=(0.556, 0.41),
    )

    decision_path = summarize_v5_screen(
        baseline_path=baseline,
        candidate_paths={"beta_005": candidate},
        candidate_betas={"beta_005": 0.05},
        output_path=tmp_path / "decision.json",
    )
    decision = json.loads(decision_path.read_text(encoding="utf-8"))

    assert decision["decision"] == "pass_v5"
    assert decision["selected"] == "beta_005"
    assert decision["evidence_scope"] == "validation_only"
    assert decision["candidate_configs"]["beta_005"]["asr_consistency_weight"] == 0.05
    assert decision["test_set_used"] is False

    selection = freeze_from_decision(decision_path, tmp_path / "selection.json")
    frozen = json.loads(selection.read_text(encoding="utf-8"))
    assert frozen["state"] == "frozen"
    assert frozen["candidate_config"]["asr_consistency_weight"] == 0.05


def test_v5_freeze_rejects_failed_screen(tmp_path: Path) -> None:
    decision = tmp_path / "decision.json"
    decision.write_text(
        json.dumps(
            {
                "decision": "stop_v5",
                "selected": None,
                "candidate_configs": {},
                "evidence_scope": "validation_only",
                "test_set_used": False,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="did not pass"):
        freeze_from_decision(decision, tmp_path / "selection.json")

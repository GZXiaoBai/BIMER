from __future__ import annotations

import json

import pytest

from bimer.v5_protocol import (
    freeze_v5_selection,
    run_guarded_v5_test,
    select_v5_candidate,
)


def _report(weighted: float, macro: float) -> dict[str, dict[str, float]]:
    return {
        "meld": {"weighted_f1": weighted, "macro_f1": macro},
        "emotiontalk": {"weighted_f1": weighted, "macro_f1": macro},
    }


def _conditions(
    *,
    clean: tuple[float, float] = (0.60, 0.45),
    whisper: tuple[float, float] = (0.55, 0.40),
    audio: tuple[float, float] = (0.57, 0.42),
    video: tuple[float, float] = (0.56, 0.41),
) -> dict[str, dict[str, dict[str, float]]]:
    return {
        "clean": _report(*clean),
        "whisper": _report(*whisper),
        "audio_10db": _report(*audio),
        "video_drop_50": _report(*video),
    }


def test_v5_selection_requires_every_predeclared_validation_gate() -> None:
    baseline = _conditions()
    accepted = _conditions(
        clean=(0.598, 0.448),
        whisper=(0.57, 0.42),
        audio=(0.566, 0.42),
        video=(0.556, 0.41),
    )
    failed_language = _conditions(whisper=(0.57, 0.42))
    failed_language["whisper"]["meld"]["weighted_f1"] = 0.552
    failed_language["whisper"]["emotiontalk"]["weighted_f1"] = 0.588

    decision = select_v5_candidate(
        baseline=baseline,
        candidates={
            "beta_005": {"beta": 0.05, "conditions": accepted},
            "beta_010": {"beta": 0.10, "conditions": failed_language},
        },
    )

    assert decision.decision == "pass_v5"
    assert decision.selected == "beta_005"
    assert decision.passed == ("beta_005",)
    assert decision.diagnostics["beta_005"]["whisper_weighted_f1_gain"] >= 0.015
    assert not decision.diagnostics["beta_010"]["accepted"]


def test_v5_selection_stops_when_no_candidate_passes() -> None:
    decision = select_v5_candidate(
        baseline=_conditions(),
        candidates={
            "beta_005": {
                "beta": 0.05,
                "conditions": _conditions(whisper=(0.56, 0.41)),
            }
        },
    )
    assert decision.decision == "stop_v5"
    assert decision.selected is None


def test_v5_freeze_requires_validation_only_evidence_and_one_test_attempt(tmp_path) -> None:
    selection = freeze_v5_selection(
        tmp_path / "selection.json",
        selected_candidate="beta_005",
        candidate_config={"asr_consistency_weight": 0.05},
        evidence={"validation_only": True, "test_set_used": False},
    )
    marker = tmp_path / "TEST_EVALUATED"
    assert run_guarded_v5_test(selection, marker, lambda: {"weighted_f1": 0.61}) == {
        "weighted_f1": 0.61
    }
    assert json.loads(marker.read_text(encoding="utf-8"))["version"] == "v5"
    with pytest.raises(RuntimeError, match="already been evaluated"):
        run_guarded_v5_test(selection, marker, lambda: None)

    with pytest.raises(ValueError, match="validation-only"):
        freeze_v5_selection(
            tmp_path / "bad.json",
            selected_candidate="beta_010",
            candidate_config={},
            evidence={"validation_only": False, "test_set_used": True},
        )

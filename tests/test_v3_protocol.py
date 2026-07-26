import json

import pytest

from bimer.v3_protocol import (
    freeze_v3_selection,
    run_guarded_v3_test,
    select_classification_loss,
    select_gate_ranking_weight,
)


def _clean(weighted=0.60, macro=0.40):
    return {
        "meld": {"weighted_f1": weighted, "macro_f1": macro},
        "emotiontalk": {"weighted_f1": weighted, "macro_f1": macro},
    }


def test_loss_selection_applies_macro_and_weighted_f1_guards():
    decision = select_classification_loss(
        baseline_name="weighted_ce",
        baseline=_clean(),
        candidates={
            "balanced_softmax": _clean(weighted=0.596, macro=0.408),
            "focal": {
                "meld": {"weighted_f1": 0.589, "macro_f1": 0.42},
                "emotiontalk": {"weighted_f1": 0.61, "macro_f1": 0.42},
            },
        },
    )

    assert decision.selected == "balanced_softmax"
    assert decision.passed == ("balanced_softmax",)


def test_gate_selection_requires_clean_robustness_and_gate_direction():
    baseline_perturbed = {
        "audio_10db": _clean(0.50),
        "video_50": _clean(0.52),
        "whisper": _clean(0.54),
    }
    passing = {
        "clean": _clean(0.598),
        "perturbed": {
            "audio_10db": _clean(0.505),
            "video_50": _clean(0.53),
            "whisper": _clean(0.54),
        },
        "gate_deltas": {
            "audio": {"mean": -0.06, "ci95": [-0.08, -0.02]},
            "vision": {"mean": -0.07, "ci95": [-0.09, -0.01]},
            "text": {"mean": -0.01, "ci95": [-0.03, 0.01]},
        },
    }
    failing_audio = json.loads(json.dumps(passing))
    failing_audio["perturbed"]["audio_10db"] = _clean(0.49)

    decision = select_gate_ranking_weight(
        baseline_clean=_clean(),
        baseline_perturbed=baseline_perturbed,
        candidates={0.05: passing, 0.10: failing_audio},
    )

    assert decision.selected == 0.05
    assert decision.passed == (0.05,)


def test_frozen_selection_allows_one_successful_test_only(tmp_path):
    selection = freeze_v3_selection(
        tmp_path / "selection.json",
        classification_loss="balanced_softmax",
        gate_ranking_weight=0.05,
        evidence={"validation_only": True},
    )
    marker = tmp_path / "TEST_EVALUATED"
    calls = []

    result = run_guarded_v3_test(
        selection,
        marker,
        lambda: calls.append("evaluated") or {"weighted_f1": 0.6},
    )

    assert result == {"weighted_f1": 0.6}
    assert marker.exists()
    with pytest.raises(RuntimeError, match="already been evaluated"):
        run_guarded_v3_test(selection, marker, lambda: None)
    assert calls == ["evaluated"]


def test_failed_test_does_not_write_evaluated_marker(tmp_path):
    selection = freeze_v3_selection(
        tmp_path / "selection.json",
        classification_loss="weighted_ce",
        gate_ranking_weight=0.0,
        evidence={},
    )
    marker = tmp_path / "TEST_EVALUATED"

    with pytest.raises(ValueError, match="boom"):
        run_guarded_v3_test(
            selection,
            marker,
            lambda: (_ for _ in ()).throw(ValueError("boom")),
        )

    assert not marker.exists()

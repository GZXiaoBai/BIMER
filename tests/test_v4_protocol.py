import json

import pytest

from bimer.v4_protocol import (
    freeze_v4_selection,
    run_guarded_v4_test,
    select_v4_candidate,
)


def _metrics(
    weighted=0.60,
    macro=0.45,
    *,
    fear=0.20,
    disgust=0.10,
    sadness=0.30,
):
    return {
        dataset: {
            "weighted_f1": weighted,
            "macro_f1": macro,
            "per_class_f1": {
                "neutral": 0.70,
                "joy": 0.60,
                "sadness": sadness,
                "anger": 0.50,
                "surprise": 0.40,
                "fear": fear,
                "disgust": disgust,
            },
        }
        for dataset in ("meld", "emotiontalk")
    }


def _evidence(*, classes=5, finite=True, missing_finite=True):
    return {
        "predicted_class_count": classes,
        "finite": finite,
        "missing_modality_finite": missing_finite,
    }


def test_v4_selection_requires_weighted_macro_minority_and_safety_gates():
    decision = select_v4_candidate(
        baseline=_metrics(),
        candidates={
            "context_only": {
                "metrics": _metrics(
                    weighted=0.611,
                    macro=0.459,
                    fear=0.22,
                    disgust=0.12,
                    sadness=0.32,
                ),
                "evidence": _evidence(),
                "complexity_rank": 1,
                "prototype_weight": 0.0,
            },
            "too_few_classes": {
                "metrics": _metrics(
                    weighted=0.62,
                    macro=0.47,
                    fear=0.24,
                    disgust=0.14,
                    sadness=0.34,
                ),
                "evidence": _evidence(classes=3),
                "complexity_rank": 2,
                "prototype_weight": 0.1,
            },
        },
    )

    assert decision.decision == "pass_v4a"
    assert decision.selected == "context_only"
    assert decision.passed == ("context_only",)
    assert decision.diagnostics["context_only"]["minority_f1_gain"] >= 0.015
    assert not decision.diagnostics["too_few_classes"]["accepted"]


def test_v4_selection_triggers_lora_and_keeps_best_candidate_when_none_pass():
    decision = select_v4_candidate(
        baseline=_metrics(),
        candidates={
            "context_only": {
                "metrics": _metrics(weighted=0.605, macro=0.454),
                "evidence": _evidence(),
                "complexity_rank": 1,
                "prototype_weight": 0.0,
            },
            "combined_mu_005": {
                "metrics": _metrics(weighted=0.607, macro=0.456),
                "evidence": _evidence(),
                "complexity_rank": 2,
                "prototype_weight": 0.05,
            },
        },
    )

    assert decision.decision == "trigger_lora"
    assert decision.selected is None
    assert decision.best_candidate == "combined_mu_005"
    assert decision.passed == ()


def test_v4_tie_prefers_simpler_candidate_with_lower_prototype_weight():
    decision = select_v4_candidate(
        baseline=_metrics(),
        candidates={
            "combined_mu_010": {
                "metrics": _metrics(
                    weighted=0.612,
                    macro=0.46,
                    fear=0.22,
                    disgust=0.12,
                    sadness=0.32,
                ),
                "evidence": _evidence(),
                "complexity_rank": 2,
                "prototype_weight": 0.10,
            },
            "context_only": {
                "metrics": _metrics(
                    weighted=0.6115,
                    macro=0.4604,
                    fear=0.22,
                    disgust=0.12,
                    sadness=0.32,
                ),
                "evidence": _evidence(),
                "complexity_rank": 1,
                "prototype_weight": 0.0,
            },
        },
    )

    assert decision.selected == "context_only"


def test_frozen_v4_selection_allows_one_exploratory_test(tmp_path):
    selection = freeze_v4_selection(
        tmp_path / "selection.json",
        selected_candidate="combined_mu_010",
        candidate_config={
            "prototype_loss_weight": 0.10,
            "use_adaptive_context_gate": True,
        },
        evidence={"validation_only": True, "test_set_used": False},
    )
    marker = tmp_path / "TEST_EVALUATED"
    calls = []

    result = run_guarded_v4_test(
        selection,
        marker,
        lambda: calls.append("evaluated") or {"weighted_f1": 0.62},
    )

    assert result == {"weighted_f1": 0.62}
    assert marker.exists()
    with pytest.raises(RuntimeError, match="already been evaluated"):
        run_guarded_v4_test(selection, marker, lambda: None)
    assert calls == ["evaluated"]


def test_v4_test_guard_rejects_non_frozen_selection(tmp_path):
    selection = tmp_path / "selection.json"
    selection.write_text('{"state": "screening", "version": "v4"}', encoding="utf-8")

    with pytest.raises(RuntimeError, match="not frozen"):
        run_guarded_v4_test(selection, tmp_path / "TEST_EVALUATED", lambda: None)


def test_failed_v4_official_test_attempt_is_still_consumed(tmp_path):
    selection = freeze_v4_selection(
        tmp_path / "selection.json",
        selected_candidate="context_only",
        candidate_config={"prototype_loss_weight": 0.0},
        evidence={"validation_only": True, "test_set_used": False},
    )
    marker = tmp_path / "TEST_EVALUATED"

    with pytest.raises(RuntimeError, match="evaluation failed"):
        run_guarded_v4_test(
            selection,
            marker,
            lambda: (_ for _ in ()).throw(RuntimeError("evaluation failed")),
        )

    assert json.loads(marker.read_text(encoding="utf-8"))["status"] == "failed"
    with pytest.raises(RuntimeError, match="already been evaluated"):
        run_guarded_v4_test(selection, marker, lambda: None)

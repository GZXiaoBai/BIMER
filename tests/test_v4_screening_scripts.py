import json

import numpy as np
import pytest

from scripts.freeze_v4_selection import freeze_from_decision
from scripts.summarize_v4_screen import summarize_screen


def _metrics(weighted, macro, minority):
    return {
        dataset: {
            "weighted_f1": weighted,
            "macro_f1": macro,
            "per_class_f1": {
                "neutral": 0.70,
                "joy": 0.60,
                "sadness": minority,
                "anger": 0.50,
                "surprise": 0.40,
                "fear": minority,
                "disgust": minority,
            },
        }
        for dataset in ("meld", "emotiontalk")
    }


def _write_candidate(root, *, weighted, macro, minority):
    result = root / "adaptive_context_prototype" / "joint" / "seed-42" / "results.json"
    result.parent.mkdir(parents=True)
    result.write_text(
        json.dumps({"validation": _metrics(weighted, macro, minority)}),
        encoding="utf-8",
    )
    for dataset in ("meld", "emotiontalk"):
        prediction_root = result.parent / "validation_predictions"
        prediction_root.mkdir(exist_ok=True)
        np.savez_compressed(
            prediction_root / f"{dataset}.npz",
            prediction=np.asarray([0, 1, 2, 3]),
            probabilities=np.full((4, 7), 1 / 7, dtype=np.float32),
            gates=np.full((4, 3), 1 / 3, dtype=np.float32),
            context_gates=np.full(4, 0.5, dtype=np.float32),
            prototype_logits=np.ones((4, 7), dtype=np.float32),
        )
    for modality in ("text", "audio", "vision"):
        condition = result.parent / "validation_conditions" / f"missing_{modality}.json"
        condition.parent.mkdir(exist_ok=True)
        condition.write_text(
            json.dumps({"validation": _metrics(weighted, macro, minority)}),
            encoding="utf-8",
        )
        prediction_root = condition.parent / f"{condition.stem}.predictions"
        prediction_root.mkdir(exist_ok=True)
        for dataset in ("meld", "emotiontalk"):
            np.savez_compressed(
                prediction_root / f"{dataset}.npz",
                probabilities=np.full((4, 7), 1 / 7, dtype=np.float32),
                gates=np.full((4, 3), 1 / 3, dtype=np.float32),
            )
    return result


def test_v4_screen_summary_and_freeze_use_validation_evidence_only(tmp_path):
    baseline = _write_candidate(
        tmp_path / "baseline",
        weighted=0.60,
        macro=0.45,
        minority=0.20,
    )
    candidate = _write_candidate(
        tmp_path / "combined_mu_010",
        weighted=0.612,
        macro=0.46,
        minority=0.22,
    )
    decision_path = summarize_screen(
        baseline_path=baseline,
        candidate_paths={"combined_mu_100": candidate},
        output_path=tmp_path / "decision.json",
    )
    decision = json.loads(decision_path.read_text(encoding="utf-8"))

    assert decision["decision"] == "pass_v4a"
    assert decision["selected"] == "combined_mu_100"
    assert decision["evidence_scope"] == "validation_only"
    selection = freeze_from_decision(decision_path, tmp_path / "selection.json")
    frozen = json.loads(selection.read_text(encoding="utf-8"))
    assert frozen["state"] == "frozen"
    assert frozen["candidate_config"]["prototype_loss_weight"] == pytest.approx(0.10)


def test_freeze_rejects_screen_decision_that_triggered_lora(tmp_path):
    decision = tmp_path / "decision.json"
    decision.write_text(
        json.dumps(
            {
                "decision": "trigger_lora",
                "selected": None,
                "candidate_configs": {},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="did not pass"):
        freeze_from_decision(decision, tmp_path / "selection.json")

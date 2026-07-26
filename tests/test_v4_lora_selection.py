import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "summarize_v4_lora.py"
LABELS = ("neutral", "joy", "sadness", "anger", "surprise", "fear", "disgust")


def _metrics(weighted: float, macro: float, minority: float):
    return {
        dataset: {
            "weighted_f1": weighted,
            "macro_f1": macro,
            "accuracy": weighted,
            "per_class_f1": {
                label: minority if label in {"fear", "disgust", "sadness"} else 0.5
                for label in LABELS
            },
        }
        for dataset in ("meld", "emotiontalk")
    }


def _candidate(root: Path, tag: str, *, weighted: float, macro: float, minority: float):
    candidate = root / tag
    run = candidate / "fusion" / "adaptive_context_prototype" / "joint" / "seed-42"
    predictions = run / "validation_predictions"
    predictions.mkdir(parents=True)
    (run / "results.json").write_text(
        json.dumps({"validation": _metrics(weighted, macro, minority), "test": {}}),
        encoding="utf-8",
    )
    for dataset in ("meld", "emotiontalk"):
        np.savez_compressed(
            predictions / f"{dataset}.npz",
            prediction=np.asarray([0, 1, 2, 3, 4]),
            probabilities=np.full((5, 7), 1 / 7),
            gates=np.full((5, 3), 1 / 3),
            context_gates=np.full(5, 0.5),
            prototype_logits=np.zeros((5, 7)),
        )
    condition_root = run / "validation_conditions"
    condition_root.mkdir()
    for modality in ("text", "audio", "vision"):
        condition = condition_root / f"missing_{modality}.json"
        condition.write_text(
            json.dumps({"validation": _metrics(weighted, macro, minority)}),
            encoding="utf-8",
        )
        condition_predictions = condition_root / f"missing_{modality}.predictions"
        condition_predictions.mkdir()
        for dataset in ("meld", "emotiontalk"):
            np.savez_compressed(
                condition_predictions / f"{dataset}.npz",
                probabilities=np.full((5, 7), 1 / 7),
                gates=np.full((5, 3), 1 / 3),
            )
    adaptation = candidate / "text-adaptation"
    adaptation.mkdir()
    (adaptation / "result.json").write_text(
        json.dumps(
            {
                "adapter_path": str(adaptation / "adapter"),
                "adapter_sha256": f"sha-{tag}",
                "base_model": "xlm-roberta-base",
            }
        ),
        encoding="utf-8",
    )
    features = candidate / "features"
    features.mkdir()
    (features / "TEXT_FEATURES_READY.json").write_text("{}", encoding="utf-8")


def test_lora_summary_applies_original_validation_gates_and_records_adapter(tmp_path):
    baseline = tmp_path / "baseline.json"
    baseline.write_text(
        json.dumps({"validation": _metrics(0.60, 0.45, 0.20)}),
        encoding="utf-8",
    )
    lora = tmp_path / "lora"
    _candidate(lora, "lr_100", weighted=0.612, macro=0.46, minority=0.22)
    _candidate(lora, "lr_200", weighted=0.605, macro=0.454, minority=0.205)
    structure = tmp_path / "structure.json"
    structure.write_text(
        json.dumps(
            {
                "candidate_config": {
                    "model": "adaptive_context_prototype",
                    "prototype_loss_weight": 0.1,
                    "use_adaptive_context_gate": True,
                }
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "decision.json"

    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--baseline",
            str(baseline),
            "--lora-root",
            str(lora),
            "--structure",
            str(structure),
            "--output",
            str(output),
        ],
        cwd=ROOT,
        check=True,
    )

    decision = json.loads(output.read_text(encoding="utf-8"))
    assert decision["decision"] == "pass_lora"
    assert decision["selected"] == "lr_100"
    assert decision["candidate_configs"]["lr_100"]["adapter_sha256"] == "sha-lr_100"
    assert decision["candidate_configs"]["lr_100"]["feature_root"].endswith("lr_100/features")

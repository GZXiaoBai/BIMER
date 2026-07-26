#!/usr/bin/env python3
# ruff: noqa: E402
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bimer.v4_protocol import select_v4_candidate

DATASETS = ("meld", "emotiontalk")
TAGS = ("lr_100", "lr_200")
MODEL = "adaptive_context_prototype"


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def _run_root(root: Path, tag: str) -> Path:
    return root / tag / "fusion" / MODEL / "joint" / "seed-42"


def _finite(path: Path, fields: tuple[str, ...]) -> bool:
    if not path.is_file():
        return False
    with np.load(path, allow_pickle=False) as payload:
        return all(field in payload.files and np.isfinite(payload[field]).all() for field in fields)


def _evidence(root: Path, tag: str) -> dict[str, object]:
    run = _run_root(root, tag)
    counts = []
    finite = True
    for dataset in DATASETS:
        path = run / "validation_predictions" / f"{dataset}.npz"
        if not path.is_file():
            return {
                "predicted_class_count": 0,
                "finite": False,
                "missing_modality_finite": False,
            }
        with np.load(path, allow_pickle=False) as payload:
            counts.append(len(np.unique(payload["prediction"])))
        finite = finite and _finite(
            path,
            ("probabilities", "gates", "context_gates", "prototype_logits"),
        )
    missing_finite = True
    conditions = run / "validation_conditions"
    for modality in ("text", "audio", "vision"):
        result_path = conditions / f"missing_{modality}.json"
        if not result_path.is_file():
            missing_finite = False
            break
        report = json.loads(result_path.read_text(encoding="utf-8"))
        prediction_root = conditions / f"missing_{modality}.predictions"
        missing_finite = (
            missing_finite
            and set(report.get("validation", {})) == set(DATASETS)
            and all(
                _finite(
                    prediction_root / f"{dataset}.npz",
                    ("probabilities", "gates"),
                )
                for dataset in DATASETS
            )
        )
    return {
        "predicted_class_count": min(counts),
        "finite": finite,
        "missing_modality_finite": missing_finite,
    }


def summarize(
    *,
    baseline: Path,
    lora_root: Path,
    structure: Path,
    output: Path,
) -> Path:
    baseline_metrics = json.loads(baseline.read_text(encoding="utf-8"))["validation"]
    structure_payload = json.loads(structure.read_text(encoding="utf-8"))
    base_config = structure_payload["candidate_config"]
    candidates = {}
    configs = {}
    for tag, learning_rate in zip(TAGS, (1e-4, 2e-4), strict=True):
        run = _run_root(lora_root, tag)
        metrics = json.loads((run / "results.json").read_text(encoding="utf-8"))["validation"]
        adaptation = json.loads(
            (lora_root / tag / "text-adaptation" / "result.json").read_text(encoding="utf-8")
        )
        config = {
            **base_config,
            "feature_root": str(lora_root / tag / "features"),
            "lora_learning_rate": learning_rate,
            "adapter_path": adaptation["adapter_path"],
            "adapter_sha256": adaptation["adapter_sha256"],
            "adapter_base_model": adaptation["base_model"],
            "complexity_rank": 3,
        }
        configs[tag] = config
        candidates[tag] = {
            "metrics": metrics,
            "evidence": _evidence(lora_root, tag),
            "complexity_rank": 3,
            "prototype_weight": float(config["prototype_loss_weight"]),
        }
    decision = select_v4_candidate(
        baseline=baseline_metrics,
        candidates=candidates,
    )
    payload = {
        **asdict(decision),
        "decision": "pass_lora" if decision.selected else "stop_v4",
        "evidence_scope": "validation_only",
        "test_set_used": False,
        "baseline": str(baseline),
        "candidate_configs": configs,
    }
    _atomic_json(output, payload)
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--lora-root", type=Path, required=True)
    parser.add_argument("--structure", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(
        summarize(
            baseline=args.baseline,
            lora_root=args.lora_root,
            structure=args.structure,
            output=args.output,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

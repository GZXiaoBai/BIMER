#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import tempfile
from dataclasses import asdict
from pathlib import Path

import numpy as np

from bimer.v4_protocol import select_v4_candidate

DATASETS = ("meld", "emotiontalk")


def _validation(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))["validation"]


def _candidate_config(name: str) -> dict[str, object]:
    if name == "context_only":
        return {
            "model": "adaptive_context_prototype",
            "prototype_loss_weight": 0.0,
            "use_adaptive_context_gate": True,
            "complexity_rank": 1,
        }
    if name == "prototype_only":
        return {
            "model": "adaptive_context_prototype",
            "prototype_loss_weight": 0.10,
            "use_adaptive_context_gate": False,
            "complexity_rank": 1,
        }
    prefix = "combined_mu_"
    if name.startswith(prefix):
        weight = int(name.removeprefix(prefix)) / 1000.0
        if weight not in {0.05, 0.10, 0.20}:
            raise ValueError(f"unsupported screened prototype weight in {name}")
        return {
            "model": "adaptive_context_prototype",
            "prototype_loss_weight": weight,
            "use_adaptive_context_gate": True,
            "complexity_rank": 2,
        }
    raise ValueError(f"unknown V4 screen candidate {name}")


def _arrays_are_finite(path: Path, fields: tuple[str, ...]) -> bool:
    if not path.is_file():
        return False
    with np.load(path, allow_pickle=False) as payload:
        return all(field in payload.files and np.isfinite(payload[field]).all() for field in fields)


def _candidate_evidence(result_path: Path) -> dict[str, object]:
    predicted_class_counts = []
    finite = True
    for dataset in DATASETS:
        prediction_path = result_path.parent / "validation_predictions" / f"{dataset}.npz"
        if not prediction_path.is_file():
            return {
                "predicted_class_count": 0,
                "finite": False,
                "missing_modality_finite": False,
            }
        with np.load(prediction_path, allow_pickle=False) as predictions:
            predicted_class_counts.append(len(np.unique(predictions["prediction"])))
        finite = finite and _arrays_are_finite(
            prediction_path,
            ("probabilities", "gates", "context_gates", "prototype_logits"),
        )
    missing_finite = True
    for modality in ("text", "audio", "vision"):
        condition = result_path.parent / "validation_conditions" / f"missing_{modality}.json"
        if not condition.is_file() or set(_validation(condition)) != set(DATASETS):
            missing_finite = False
            break
        prediction_root = condition.parent / f"{condition.stem}.predictions"
        missing_finite = missing_finite and all(
            _arrays_are_finite(
                prediction_root / f"{dataset}.npz",
                ("probabilities", "gates"),
            )
            for dataset in DATASETS
        )
    return {
        "predicted_class_count": min(predicted_class_counts),
        "finite": finite,
        "missing_modality_finite": missing_finite,
    }


def _atomic_json(path: Path, payload: object) -> Path:
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
    return path


def summarize_screen(
    *,
    baseline_path: Path,
    candidate_paths: dict[str, Path],
    output_path: Path,
) -> Path:
    candidate_configs = {name: _candidate_config(name) for name in candidate_paths}
    decision = select_v4_candidate(
        baseline=_validation(baseline_path),
        candidates={
            name: {
                "metrics": _validation(path),
                "evidence": _candidate_evidence(path),
                "complexity_rank": candidate_configs[name]["complexity_rank"],
                "prototype_weight": candidate_configs[name]["prototype_loss_weight"],
            }
            for name, path in candidate_paths.items()
        },
    )
    payload = {
        **asdict(decision),
        "evidence_scope": "validation_only",
        "test_set_used": False,
        "baseline": str(baseline_path),
        "candidates": {name: str(path) for name, path in candidate_paths.items()},
        "candidate_configs": candidate_configs,
    }
    return _atomic_json(output_path, payload)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", action="append", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    candidates = {}
    for specification in args.candidate:
        name, path = specification.split("=", 1)
        candidates[name] = Path(path)
    result = summarize_screen(
        baseline_path=Path(args.baseline),
        candidate_paths=candidates,
        output_path=Path(args.output),
    )
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

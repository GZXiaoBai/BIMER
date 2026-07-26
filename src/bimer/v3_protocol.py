from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, TypeVar

DATASETS = ("meld", "emotiontalk")
T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class SelectionDecision:
    selected: str | float
    passed: tuple
    diagnostics: dict[str, dict[str, float | bool]]


def _average(report: Mapping[str, Mapping[str, float]], metric: str) -> float:
    return sum(float(report[dataset][metric]) for dataset in DATASETS) / len(DATASETS)


def select_classification_loss(
    *,
    baseline_name: str,
    baseline: Mapping[str, Mapping[str, float]],
    candidates: Mapping[str, Mapping[str, Mapping[str, float]]],
) -> SelectionDecision:
    baseline_macro = _average(baseline, "macro_f1")
    baseline_weighted = _average(baseline, "weighted_f1")
    diagnostics: dict[str, dict[str, float | bool]] = {}
    passed: list[str] = []
    for name, report in candidates.items():
        macro_gain = _average(report, "macro_f1") - baseline_macro
        weighted_delta = _average(report, "weighted_f1") - baseline_weighted
        worst_dataset_delta = min(
            float(report[dataset]["weighted_f1"])
            - float(baseline[dataset]["weighted_f1"])
            for dataset in DATASETS
        )
        accepted = (
            macro_gain >= 0.005
            and weighted_delta >= -0.005
            and worst_dataset_delta >= -0.01
        )
        diagnostics[name] = {
            "macro_f1_gain": macro_gain,
            "weighted_f1_delta": weighted_delta,
            "worst_dataset_weighted_f1_delta": worst_dataset_delta,
            "accepted": accepted,
        }
        if accepted:
            passed.append(name)
    selected = (
        max(
            passed,
            key=lambda name: (
                _average(candidates[name], "macro_f1"),
                _average(candidates[name], "weighted_f1"),
            ),
        )
        if passed
        else baseline_name
    )
    return SelectionDecision(selected, tuple(passed), diagnostics)


def select_gate_ranking_weight(
    *,
    baseline_clean: Mapping[str, Mapping[str, float]],
    baseline_perturbed: Mapping[str, Mapping[str, Mapping[str, float]]],
    candidates: Mapping[
        float,
        Mapping[str, object],
    ],
) -> SelectionDecision:
    required_conditions = ("audio_10db", "video_50", "whisper")
    baseline_clean_score = _average(baseline_clean, "weighted_f1")
    diagnostics: dict[str, dict[str, float | bool]] = {}
    passed: list[float] = []
    robustness_scores: dict[float, float] = {}
    for weight, payload in candidates.items():
        clean = payload["clean"]
        perturbed = payload["perturbed"]
        gate_deltas = payload["gate_deltas"]
        assert isinstance(clean, Mapping)
        assert isinstance(perturbed, Mapping)
        assert isinstance(gate_deltas, Mapping)
        clean_delta = _average(clean, "weighted_f1") - baseline_clean_score
        condition_deltas = {
            condition: _average(perturbed[condition], "weighted_f1")
            - _average(baseline_perturbed[condition], "weighted_f1")
            for condition in required_conditions
        }
        robustness_gain = sum(condition_deltas.values()) / len(condition_deltas)
        audio_gate = gate_deltas["audio"]
        vision_gate = gate_deltas["vision"]
        text_gate = gate_deltas["text"]
        gate_ok = (
            float(audio_gate["mean"]) <= -0.05
            and float(audio_gate["ci95"][1]) < 0
            and float(vision_gate["mean"]) <= -0.05
            and float(vision_gate["ci95"][1]) < 0
            and float(text_gate["mean"]) < 0
        )
        accepted = (
            clean_delta >= -0.005
            and robustness_gain >= 0.005
            and condition_deltas["audio_10db"] >= 0
            and gate_ok
        )
        key = f"{float(weight):g}"
        diagnostics[key] = {
            "clean_weighted_f1_delta": clean_delta,
            "perturbed_mean_weighted_f1_gain": robustness_gain,
            "audio_10db_weighted_f1_delta": condition_deltas["audio_10db"],
            "audio_gate_delta": float(audio_gate["mean"]),
            "vision_gate_delta": float(vision_gate["mean"]),
            "text_gate_delta": float(text_gate["mean"]),
            "accepted": accepted,
        }
        robustness_scores[weight] = robustness_gain
        if accepted:
            passed.append(weight)
    selected = (
        max(passed, key=lambda weight: (robustness_scores[weight], -weight))
        if passed
        else 0.0
    )
    return SelectionDecision(selected, tuple(passed), diagnostics)


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        temporary = Path(temporary_name)
        if temporary.exists():
            temporary.unlink()


def freeze_v3_selection(
    path: Path | str,
    *,
    classification_loss: str,
    gate_ranking_weight: float,
    evidence: Mapping[str, object],
) -> Path:
    if classification_loss not in {"weighted_ce", "balanced_softmax", "focal"}:
        raise ValueError("unsupported classification loss")
    if gate_ranking_weight not in {0.0, 0.05, 0.10, 0.20}:
        raise ValueError("gate ranking weight is outside the screened grid")
    target = Path(path)
    _atomic_json(
        target,
        {
            "state": "frozen",
            "version": "v3",
            "classification_loss": classification_loss,
            "gate_ranking_weight": gate_ranking_weight,
            "evidence": dict(evidence),
        },
    )
    return target


def run_guarded_v3_test(
    selection_path: Path | str,
    marker_path: Path | str,
    evaluator: Callable[[], T],
) -> T:
    selection = json.loads(Path(selection_path).read_text(encoding="utf-8"))
    if selection.get("state") != "frozen" or selection.get("version") != "v3":
        raise RuntimeError("V3 selection configuration is not frozen")
    marker = Path(marker_path)
    if marker.exists():
        raise RuntimeError("V3 official test has already been evaluated")
    marker.parent.mkdir(parents=True, exist_ok=True)
    claim = marker.with_name(f"{marker.name}.RUNNING")
    try:
        descriptor = os.open(claim, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise RuntimeError("V3 official test evaluation is already running") from exc
    os.close(descriptor)
    try:
        result = evaluator()
        _atomic_json(
            marker,
            {
                "status": "evaluated",
                "selection": str(Path(selection_path).resolve()),
            },
        )
        return result
    finally:
        claim.unlink(missing_ok=True)

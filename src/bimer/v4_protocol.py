from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, TypeVar, cast

DATASETS = ("meld", "emotiontalk")
MINORITY_LABELS = ("fear", "disgust", "sadness")
T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class V4SelectionDecision:
    decision: str
    selected: str | None
    best_candidate: str
    passed: tuple[str, ...]
    diagnostics: dict[str, dict[str, float | bool]]


def _number(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    return float(value)


def _average_metric(report: Mapping[str, Mapping[str, object]], metric: str) -> float:
    return sum(_number(report[dataset][metric], name=metric) for dataset in DATASETS) / len(
        DATASETS
    )


def _minority_f1(report: Mapping[str, Mapping[str, object]]) -> float:
    values: list[float] = []
    for dataset in DATASETS:
        per_class = report[dataset]["per_class_f1"]
        if not isinstance(per_class, Mapping):
            raise ValueError("per_class_f1 must be a mapping")
        values.extend(
            _number(per_class[label], name=f"per_class_f1.{label}") for label in MINORITY_LABELS
        )
    return sum(values) / len(values)


def select_v4_candidate(
    *,
    baseline: Mapping[str, Mapping[str, object]],
    candidates: Mapping[str, Mapping[str, object]],
) -> V4SelectionDecision:
    if not candidates:
        raise ValueError("at least one V4 candidate is required")
    baseline_weighted = _average_metric(baseline, "weighted_f1")
    baseline_macro = _average_metric(baseline, "macro_f1")
    baseline_minority = _minority_f1(baseline)
    diagnostics: dict[str, dict[str, float | bool]] = {}
    passed: list[str] = []
    candidate_scores: dict[str, float] = {}
    for name, payload in candidates.items():
        metrics = payload["metrics"]
        evidence = payload["evidence"]
        if not isinstance(metrics, Mapping) or not isinstance(evidence, Mapping):
            raise ValueError("candidate metrics and evidence must be mappings")
        dataset_metrics = cast(Mapping[str, Mapping[str, object]], metrics)
        weighted_gain = _average_metric(dataset_metrics, "weighted_f1") - baseline_weighted
        macro_gain = _average_metric(dataset_metrics, "macro_f1") - baseline_macro
        minority_gain = _minority_f1(dataset_metrics) - baseline_minority
        worst_dataset_delta = min(
            _number(dataset_metrics[dataset]["weighted_f1"], name="weighted_f1")
            - _number(baseline[dataset]["weighted_f1"], name="weighted_f1")
            for dataset in DATASETS
        )
        finite = bool(evidence.get("finite", False))
        missing_finite = bool(evidence.get("missing_modality_finite", False))
        predicted_class_count = int(
            _number(
                evidence.get("predicted_class_count", 0),
                name="predicted_class_count",
            )
        )
        accepted = (
            weighted_gain >= 0.010
            and macro_gain >= 0.008
            and worst_dataset_delta >= -0.003
            and minority_gain >= 0.015
            and predicted_class_count >= 4
            and finite
            and missing_finite
        )
        balanced_score = 0.5 * weighted_gain + 0.5 * macro_gain
        diagnostics[name] = {
            "weighted_f1_gain": weighted_gain,
            "macro_f1_gain": macro_gain,
            "worst_dataset_weighted_f1_delta": worst_dataset_delta,
            "minority_f1_gain": minority_gain,
            "predicted_class_count": float(predicted_class_count),
            "finite": finite,
            "missing_modality_finite": missing_finite,
            "balanced_score": balanced_score,
            "accepted": accepted,
        }
        candidate_scores[name] = balanced_score
        if accepted:
            passed.append(name)

    def preferred(names: list[str]) -> str:
        best_score = max(candidate_scores[name] for name in names)
        tied = [name for name in names if best_score - candidate_scores[name] <= 0.001]
        return min(
            tied,
            key=lambda name: (
                int(
                    _number(
                        candidates[name].get("complexity_rank", 999),
                        name="complexity_rank",
                    )
                ),
                _number(
                    candidates[name].get("prototype_weight", 999.0),
                    name="prototype_weight",
                ),
                name,
            ),
        )

    best_candidate = preferred(list(candidates))
    selected = preferred(passed) if passed else None
    return V4SelectionDecision(
        decision="pass_v4a" if selected is not None else "trigger_lora",
        selected=selected,
        best_candidate=best_candidate,
        passed=tuple(passed),
        diagnostics=diagnostics,
    )


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


def freeze_v4_selection(
    path: Path | str,
    *,
    selected_candidate: str,
    candidate_config: Mapping[str, object],
    evidence: Mapping[str, object],
) -> Path:
    if not selected_candidate:
        raise ValueError("selected_candidate must not be empty")
    if evidence.get("validation_only") is not True or evidence.get("test_set_used") is not False:
        raise ValueError("V4 selection evidence must be validation-only")
    target = Path(path)
    _atomic_json(
        target,
        {
            "state": "frozen",
            "version": "v4",
            "selected_candidate": selected_candidate,
            "candidate_config": dict(candidate_config),
            "evidence": dict(evidence),
        },
    )
    return target


def run_guarded_v4_test(
    selection_path: Path | str,
    marker_path: Path | str,
    evaluator: Callable[[], T],
) -> T:
    selection = json.loads(Path(selection_path).read_text(encoding="utf-8"))
    if selection.get("state") != "frozen" or selection.get("version") != "v4":
        raise RuntimeError("V4 selection configuration is not frozen")
    marker = Path(marker_path)
    if marker.exists():
        raise RuntimeError("V4 official test has already been evaluated")
    marker.parent.mkdir(parents=True, exist_ok=True)
    claim = marker.with_name(f"{marker.name}.RUNNING")
    try:
        descriptor = os.open(claim, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise RuntimeError("V4 official test evaluation is already running") from exc
    os.close(descriptor)
    _atomic_json(
        marker,
        {
            "status": "running",
            "selection": str(Path(selection_path).resolve()),
        },
    )
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
    except Exception as exc:
        _atomic_json(
            marker,
            {
                "status": "failed",
                "selection": str(Path(selection_path).resolve()),
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
        )
        raise
    finally:
        claim.unlink(missing_ok=True)

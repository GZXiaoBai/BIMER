from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, TypeVar, cast

from .experiment_protocol import run_guarded_exploratory_test

DATASETS = ("meld", "emotiontalk")
T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class V5SelectionDecision:
    decision: str
    selected: str | None
    passed: tuple[str, ...]
    diagnostics: dict[str, dict[str, float | bool]]


def _number(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    return float(value)


def _condition(
    payload: Mapping[str, object],
    condition: str,
) -> Mapping[str, Mapping[str, object]]:
    value = payload.get(condition)
    if not isinstance(value, Mapping):
        raise ValueError(f"{condition} must be a bilingual metrics mapping")
    return cast(Mapping[str, Mapping[str, object]], value)


def _average(report: Mapping[str, Mapping[str, object]], metric: str) -> float:
    return sum(_number(report[dataset][metric], name=metric) for dataset in DATASETS) / 2


def select_v5_candidate(
    *,
    baseline: Mapping[str, object],
    candidates: Mapping[str, Mapping[str, object]],
) -> V5SelectionDecision:
    if not candidates:
        raise ValueError("at least one V5 candidate is required")
    baseline_conditions = {
        name: _condition(baseline, name)
        for name in ("clean", "whisper", "audio_10db", "video_drop_50")
    }
    diagnostics: dict[str, dict[str, float | bool]] = {}
    passed: list[str] = []
    for name, payload in candidates.items():
        conditions_value = payload.get("conditions")
        if not isinstance(conditions_value, Mapping):
            raise ValueError("candidate conditions must be a mapping")
        conditions = {
            condition: _condition(conditions_value, condition) for condition in baseline_conditions
        }
        clean_weighted_delta = _average(conditions["clean"], "weighted_f1") - _average(
            baseline_conditions["clean"], "weighted_f1"
        )
        clean_macro_delta = _average(conditions["clean"], "macro_f1") - _average(
            baseline_conditions["clean"], "macro_f1"
        )
        whisper_gain = _average(conditions["whisper"], "weighted_f1") - _average(
            baseline_conditions["whisper"], "weighted_f1"
        )
        worst_language_whisper_gain = min(
            _number(
                conditions["whisper"][dataset]["weighted_f1"],
                name="weighted_f1",
            )
            - _number(
                baseline_conditions["whisper"][dataset]["weighted_f1"],
                name="weighted_f1",
            )
            for dataset in DATASETS
        )
        audio_delta = _average(conditions["audio_10db"], "weighted_f1") - _average(
            baseline_conditions["audio_10db"], "weighted_f1"
        )
        video_delta = _average(conditions["video_drop_50"], "weighted_f1") - _average(
            baseline_conditions["video_drop_50"], "weighted_f1"
        )
        accepted = (
            clean_weighted_delta >= -0.003
            and whisper_gain >= 0.015
            and worst_language_whisper_gain >= 0.005
            and clean_macro_delta >= -0.003
            and audio_delta >= -0.005
            and video_delta >= -0.005
        )
        diagnostics[name] = {
            "clean_weighted_f1_delta": clean_weighted_delta,
            "clean_macro_f1_delta": clean_macro_delta,
            "whisper_weighted_f1_gain": whisper_gain,
            "worst_language_whisper_weighted_f1_gain": worst_language_whisper_gain,
            "audio_10db_weighted_f1_delta": audio_delta,
            "video_drop_50_weighted_f1_delta": video_delta,
            "accepted": accepted,
        }
        if accepted:
            passed.append(name)

    selected = None
    if passed:
        selected = max(
            passed,
            key=lambda name: (
                float(diagnostics[name]["whisper_weighted_f1_gain"]),
                float(diagnostics[name]["clean_weighted_f1_delta"]),
                -_number(candidates[name].get("beta", 999.0), name="beta"),
            ),
        )
    return V5SelectionDecision(
        decision="pass_v5" if selected is not None else "stop_v5",
        selected=selected,
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
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        Path(temporary_name).unlink(missing_ok=True)


def freeze_v5_selection(
    path: Path | str,
    *,
    selected_candidate: str,
    candidate_config: Mapping[str, object],
    evidence: Mapping[str, object],
) -> Path:
    if not selected_candidate:
        raise ValueError("selected_candidate must not be empty")
    if evidence.get("validation_only") is not True or evidence.get("test_set_used") is not False:
        raise ValueError("V5 selection evidence must be validation-only")
    target = Path(path)
    _atomic_json(
        target,
        {
            "state": "frozen",
            "version": "v5",
            "selected_candidate": selected_candidate,
            "candidate_config": dict(candidate_config),
            "evidence": dict(evidence),
        },
    )
    return target


def run_guarded_v5_test(
    selection_path: Path | str,
    marker_path: Path | str,
    evaluator: Callable[[], T],
) -> T:
    return run_guarded_exploratory_test(
        version="v5",
        selection_path=selection_path,
        marker_path=marker_path,
        evaluator=evaluator,
    )

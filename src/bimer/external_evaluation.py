from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from sklearn.metrics import cohen_kappa_score

from .calibration import calibration_metrics
from .metrics import classification_metrics
from .runtime_cache import RuntimeFeatureCache

EXTERNAL_CONDITIONS = (
    "normal_face",
    "no_face",
    "background_noise",
    "multi_cut",
    "accent_fast_change",
)


@dataclass(frozen=True, slots=True)
class ExternalVideo:
    video_id: str
    path: str
    sha256: str
    language: str
    condition: str
    duration_seconds: float


def validate_external_video_plan(
    videos: Sequence[ExternalVideo],
) -> dict[str, object]:
    if len(videos) != 20:
        raise ValueError("external evaluation requires exactly 20 videos")
    if len({video.video_id for video in videos}) != len(videos):
        raise ValueError("external video ids must be unique")
    for video in videos:
        if video.language not in {"en", "zh"}:
            raise ValueError("external video language must be en or zh")
        if video.condition not in EXTERNAL_CONDITIONS:
            raise ValueError("unknown external video condition")
        if not 30.0 <= video.duration_seconds <= 60.0:
            raise ValueError("external videos must be 30-60 seconds")
        if len(video.sha256) != 64 or any(
            character not in "0123456789abcdef" for character in video.sha256
        ):
            raise ValueError("external video sha256 must be a lowercase digest")
    by_language = {
        language: sum(video.language == language for video in videos) for language in ("en", "zh")
    }
    by_language_condition = {
        f"{language}:{condition}": sum(
            video.language == language and video.condition == condition for video in videos
        )
        for language in ("en", "zh")
        for condition in EXTERNAL_CONDITIONS
    }
    if by_language != {"en": 10, "zh": 10} or any(
        count != 2 for count in by_language_condition.values()
    ):
        raise ValueError("external plan must contain two videos per language and condition")
    return {
        "count": len(videos),
        "by_language": by_language,
        "by_language_condition": by_language_condition,
        "locked": True,
    }


def lock_external_video_plan(
    paths: Sequence[Path | str],
    *,
    languages: Sequence[str],
    conditions: Sequence[str],
    durations: Sequence[float],
    output_path: Path | str,
) -> Path:
    if not (len(paths) == len(languages) == len(conditions) == len(durations)):
        raise ValueError("external plan fields must have equal length")
    videos = []
    for index, path_value in enumerate(paths):
        path = Path(path_value)
        digest = RuntimeFeatureCache.file_sha256(path)
        videos.append(
            ExternalVideo(
                video_id=f"external-{index:02d}",
                path=str(path.resolve()),
                sha256=digest,
                language=languages[index],
                condition=conditions[index],
                duration_seconds=float(durations[index]),
            )
        )
    report = validate_external_video_plan(videos)
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(
            {
                "videos": [asdict(video) for video in videos],
                "validation": report,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return target


def annotation_agreement(
    first: Sequence[str],
    second: Sequence[str],
) -> dict[str, float | bool]:
    if len(first) != len(second) or not first:
        raise ValueError("annotation sequences must be non-empty and aligned")
    first_values = np.asarray(first, dtype=str)
    second_values = np.asarray(second, dtype=str)
    raw = float((first_values == second_values).mean())
    kappa = float(cohen_kappa_score(first_values, second_values))
    if not np.isfinite(kappa):
        kappa = 1.0 if raw == 1.0 else 0.0
    return {
        "raw_agreement": raw,
        "cohen_kappa": kappa,
        "requires_reannotation": bool(kappa < 0.60),
    }


def _metric_bundle(
    truth: np.ndarray,
    probabilities: np.ndarray,
    label_names: Sequence[str],
) -> dict[str, object]:
    prediction = probabilities.argmax(axis=1)
    return {
        **classification_metrics(truth, prediction, label_names=label_names),
        **calibration_metrics(probabilities, truth, bins=15),
    }


def evaluate_external_predictions(
    truth: np.ndarray,
    probabilities: np.ndarray,
    *,
    video_ids: np.ndarray,
    conditions: np.ndarray,
    label_names: Sequence[str],
    bootstrap_iterations: int = 2000,
    seed: int = 42,
) -> dict[str, object]:
    labels = np.asarray(truth, dtype=np.int64)
    scores = np.asarray(probabilities, dtype=np.float64)
    clusters = np.asarray(video_ids, dtype=str)
    condition_values = np.asarray(conditions, dtype=str)
    if not (labels.shape == clusters.shape == condition_values.shape == (len(scores),)):
        raise ValueError("external prediction arrays must align")
    if bootstrap_iterations <= 0:
        raise ValueError("bootstrap_iterations must be positive")
    overall = _metric_bundle(labels, scores, label_names)
    by_condition = {
        condition: _metric_bundle(
            labels[condition_values == condition],
            scores[condition_values == condition],
            label_names,
        )
        for condition in EXTERNAL_CONDITIONS
        if bool((condition_values == condition).any())
    }
    unique_clusters = np.unique(clusters)
    random = np.random.default_rng(seed)
    bootstrap: dict[str, list[float]] = {
        name: [] for name in ("weighted_f1", "macro_f1", "accuracy", "ece", "brier")
    }
    for _ in range(bootstrap_iterations):
        sampled = random.choice(unique_clusters, size=len(unique_clusters), replace=True)
        indices = np.concatenate([np.flatnonzero(clusters == value) for value in sampled])
        metrics = _metric_bundle(labels[indices], scores[indices], label_names)
        for name in bootstrap:
            bootstrap[name].append(float(metrics[name]))
    intervals = {
        name: [
            float(np.quantile(values, 0.025)),
            float(np.quantile(values, 0.975)),
        ]
        for name, values in bootstrap.items()
    }
    return {
        **overall,
        "by_condition": by_condition,
        "ci95": intervals,
        "bootstrap_unit": "video",
        "bootstrap_iterations": bootstrap_iterations,
    }


def v3_external_acceptance(
    v2: dict[str, object],
    v3: dict[str, object],
) -> dict[str, object]:
    overall_delta = float(v3["weighted_f1"]) - float(v2["weighted_f1"])
    v2_conditions = v2["by_condition"]
    v3_conditions = v3["by_condition"]
    condition_deltas = {
        condition: float(v3_conditions[condition]["weighted_f1"])
        - float(v2_conditions[condition]["weighted_f1"])
        for condition in EXTERNAL_CONDITIONS
    }
    condition_mean = float(np.mean(list(condition_deltas.values())))
    worst_condition = min(condition_deltas.values())
    ece_delta = float(v3["ece"]) - float(v2["ece"])
    checks = {
        "overall_within_one_point": overall_delta >= -0.01,
        "condition_mean_improves_one_point": condition_mean >= 0.01,
        "no_condition_loses_three_points": worst_condition >= -0.03,
        "ece_not_worse": ece_delta <= 0,
    }
    return {
        "accepted": all(checks.values()),
        "checks": checks,
        "overall_weighted_f1_delta": overall_delta,
        "condition_mean_weighted_f1_delta": condition_mean,
        "worst_condition_weighted_f1_delta": worst_condition,
        "ece_delta": ece_delta,
        "condition_deltas": condition_deltas,
    }

import numpy as np
import pytest

from bimer.external_evaluation import (
    ExternalVideo,
    annotation_agreement,
    evaluate_external_predictions,
    v3_external_acceptance,
    validate_external_video_plan,
)

CONDITIONS = (
    "normal_face",
    "no_face",
    "background_noise",
    "multi_cut",
    "accent_fast_change",
)


def _plan():
    return [
        ExternalVideo(
            video_id=f"{language}-{condition}-{index}",
            path=f"{language}-{condition}-{index}.mp4",
            sha256=f"{len(language)}{len(condition)}{index}".ljust(64, "0"),
            language=language,
            condition=condition,
            duration_seconds=45.0,
        )
        for language in ("en", "zh")
        for condition in CONDITIONS
        for index in range(2)
    ]


def test_external_plan_requires_twenty_balanced_locked_videos():
    report = validate_external_video_plan(_plan())

    assert report["count"] == 20
    assert report["by_language"] == {"en": 10, "zh": 10}
    assert all(count == 2 for count in report["by_language_condition"].values())
    with pytest.raises(ValueError, match="exactly 20"):
        validate_external_video_plan(_plan()[:-1])


def test_annotation_agreement_reports_raw_agreement_and_kappa():
    first = ["neutral", "joy", "sadness", "anger"] * 5
    second = list(first)
    second[-1] = "neutral"

    report = annotation_agreement(first, second)

    assert report["raw_agreement"] == 0.95
    assert report["cohen_kappa"] > 0.90
    assert report["requires_reannotation"] is False


def test_external_evaluation_and_v3_acceptance_use_video_clusters():
    truth = np.asarray([0, 1] * 10)
    probabilities = np.tile(np.asarray([[0.8, 0.2], [0.2, 0.8]]), (10, 1))
    video_ids = np.asarray([f"v{index // 2}" for index in range(20)])
    conditions = np.asarray([CONDITIONS[index % 5] for index in range(20)])
    report = evaluate_external_predictions(
        truth,
        probabilities,
        video_ids=video_ids,
        conditions=conditions,
        label_names=("neutral", "joy"),
        bootstrap_iterations=50,
        seed=42,
    )

    assert report["weighted_f1"] == 1.0
    assert report["bootstrap_unit"] == "video"
    assert set(report["by_condition"]) == set(CONDITIONS)

    v2 = {
        "weighted_f1": 0.60,
        "ece": 0.10,
        "by_condition": {condition: {"weighted_f1": 0.50} for condition in CONDITIONS},
    }
    v3 = {
        "weighted_f1": 0.60,
        "ece": 0.09,
        "by_condition": {condition: {"weighted_f1": 0.52} for condition in CONDITIONS},
    }
    acceptance = v3_external_acceptance(v2, v3)
    assert acceptance["accepted"] is True

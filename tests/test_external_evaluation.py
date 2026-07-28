import json
from pathlib import Path

import numpy as np
import pytest

from bimer.external_evaluation import (
    ExternalVideo,
    annotation_agreement,
    evaluate_external_predictions,
    external_model_acceptance,
    lock_external_video_plan,
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
            authorization_basis="self_recorded",
            authorization_reference=f"consent-{language}-{condition}-{index}",
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
    invalid = _plan()
    invalid[0] = ExternalVideo(
        video_id=invalid[0].video_id,
        path=invalid[0].path,
        sha256=invalid[0].sha256,
        language=invalid[0].language,
        condition=invalid[0].condition,
        duration_seconds=invalid[0].duration_seconds,
        authorization_basis="",
        authorization_reference="",
    )
    with pytest.raises(ValueError, match="authorization"):
        validate_external_video_plan(invalid)


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
    assert external_model_acceptance(v2, v3) == acceptance


def test_lock_external_video_plan_hashes_and_validates_all_inputs(
    tmp_path: Path,
) -> None:
    paths = []
    languages = []
    conditions = []
    bases = []
    references = []
    for index, video in enumerate(_plan()):
        path = tmp_path / f"{index:02d}.mp4"
        path.write_bytes(f"video-{index}".encode())
        paths.append(path)
        languages.append(video.language)
        conditions.append(video.condition)
        bases.append(video.authorization_basis)
        references.append(video.authorization_reference)

    output = lock_external_video_plan(
        paths,
        languages=languages,
        conditions=conditions,
        durations=[45.0] * 20,
        authorization_bases=bases,
        authorization_references=references,
        output_path=tmp_path / "locked" / "plan.json",
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["validation"]["locked"] is True
    assert len(payload["videos"]) == 20
    assert all(len(video["sha256"]) == 64 for video in payload["videos"])

    with pytest.raises(ValueError, match="equal length"):
        lock_external_video_plan(
            paths,
            languages=languages[:-1],
            conditions=conditions,
            durations=[45.0] * 20,
            authorization_bases=bases,
            authorization_references=references,
            output_path=tmp_path / "invalid.json",
        )


@pytest.mark.parametrize(
    ("replacement", "message"),
    [
        ({"language": "fr"}, "language"),
        ({"condition": "studio"}, "condition"),
        ({"authorization_basis": "unknown"}, "authorization basis"),
        ({"authorization_reference": "  "}, "reference"),
        ({"duration_seconds": 29.0}, "30-60"),
        ({"sha256": "ABC"}, "lowercase digest"),
    ],
)
def test_external_plan_rejects_invalid_metadata(replacement, message):
    plan = _plan()
    original = plan[0]
    values = {
        "video_id": original.video_id,
        "path": original.path,
        "sha256": original.sha256,
        "language": original.language,
        "condition": original.condition,
        "duration_seconds": original.duration_seconds,
        "authorization_basis": original.authorization_basis,
        "authorization_reference": original.authorization_reference,
    }
    values.update(replacement)
    plan[0] = ExternalVideo(**values)

    with pytest.raises(ValueError, match=message):
        validate_external_video_plan(plan)


def test_external_plan_rejects_duplicate_ids_and_unbalanced_conditions():
    duplicate = _plan()
    duplicate[1] = ExternalVideo(
        **{
            field: (duplicate[0].video_id if field == "video_id" else getattr(duplicate[1], field))
            for field in ExternalVideo.__dataclass_fields__
        }
    )
    with pytest.raises(ValueError, match="unique"):
        validate_external_video_plan(duplicate)

    unbalanced = _plan()
    first = unbalanced[0]
    unbalanced[0] = ExternalVideo(
        **{
            field: ("background_noise" if field == "condition" else getattr(first, field))
            for field in ExternalVideo.__dataclass_fields__
        }
    )
    with pytest.raises(ValueError, match="two videos"):
        validate_external_video_plan(unbalanced)


def test_external_annotation_and_prediction_inputs_are_validated():
    with pytest.raises(ValueError, match="non-empty"):
        annotation_agreement([], [])
    with pytest.raises(ValueError, match="aligned"):
        annotation_agreement(["joy"], ["joy", "sadness"])

    truth = np.asarray([0, 1])
    probabilities = np.asarray([[0.8, 0.2], [0.2, 0.8]])
    with pytest.raises(ValueError, match="align"):
        evaluate_external_predictions(
            truth,
            probabilities,
            video_ids=np.asarray(["v0"]),
            conditions=np.asarray(["normal_face", "normal_face"]),
            label_names=("neutral", "joy"),
        )
    with pytest.raises(ValueError, match="positive"):
        evaluate_external_predictions(
            truth,
            probabilities,
            video_ids=np.asarray(["v0", "v1"]),
            conditions=np.asarray(["normal_face", "normal_face"]),
            label_names=("neutral", "joy"),
            bootstrap_iterations=0,
        )

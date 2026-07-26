import json

import numpy as np

from bimer.feature_statistics import (
    compute_feature_statistics,
    write_feature_statistics,
)
from bimer.feature_store import FeatureShard, FeatureStore
from bimer.schema import UtteranceRecord


def _record(index: int, emotion: str) -> UtteranceRecord:
    return UtteranceRecord(
        dataset="emotiontalk",
        split="train",
        dialogue_id="dialogue-1",
        utterance_id=index,
        text=f"line {index}",
        emotion=emotion,
        language="zh",
        start_seconds=float(index),
        end_seconds=float(index + 1),
    )


def test_feature_statistics_reports_labels_availability_and_dimensions(tmp_path):
    records = [_record(0, "neutral"), _record(1, "joy")]
    store = FeatureStore(tmp_path / "features")
    store.write(
        "emotiontalk",
        "train",
        0,
        FeatureShard(
            sample_ids=np.asarray([record.sample_id for record in records]),
            text=np.asarray([[3.0, 4.0, 0.0, 0.0], [0.0, 0.0, 0.0, 2.0]], np.float32),
            audio=np.ones((2, 6), np.float32),
            vision=np.asarray([[1.0] * 5, [0.0] * 5], np.float32),
            modality_mask=np.asarray([[True, True, True], [True, True, False]]),
        ),
    )

    report = compute_feature_statistics(
        records,
        store,
        dataset="emotiontalk",
        split="train",
        expected_dims={"text": 4, "audio": 6, "vision": 5},
    )

    assert report["sample_count"] == 2
    assert report["shard_count"] == 1
    assert report["label_counts"]["neutral"] == 1
    assert report["label_counts"]["joy"] == 1
    assert report["modalities"]["vision"]["available_count"] == 1
    assert report["modalities"]["vision"]["unavailable_count"] == 1
    assert report["modalities"]["text"]["dimension"] == 4
    assert report["modalities"]["text"]["mean_l2_norm"] == 3.5
    assert report["modalities"]["vision"]["available_zero_vector_count"] == 0
    assert report["modalities"]["vision"]["nonfinite_row_count"] == 0
    assert report["missing_manifest_samples"] == 0
    assert report["unexpected_feature_samples"] == 0


def test_feature_statistics_reports_missing_and_unexpected_sample_ids(tmp_path):
    records = [_record(0, "neutral"), _record(1, "joy")]
    store = FeatureStore(tmp_path / "features")
    store.write(
        "emotiontalk",
        "train",
        0,
        FeatureShard(
            sample_ids=np.asarray([records[0].sample_id, "emotiontalk:train:unexpected:99"]),
            text=np.ones((2, 4), np.float32),
            audio=np.ones((2, 6), np.float32),
            vision=np.ones((2, 5), np.float32),
            modality_mask=np.ones((2, 3), np.bool_),
        ),
    )

    report = compute_feature_statistics(
        records,
        store,
        dataset="emotiontalk",
        split="train",
        expected_dims={"text": 4, "audio": 6, "vision": 5},
    )

    assert report["missing_manifest_samples"] == 1
    assert report["unexpected_feature_samples"] == 1


def test_write_feature_statistics_creates_utf8_json(tmp_path):
    output = write_feature_statistics(
        {"dataset": "emotiontalk", "split": "train", "note": "中文"},
        tmp_path / "reports" / "stats.json",
    )

    assert json.loads(output.read_text(encoding="utf-8"))["note"] == "中文"

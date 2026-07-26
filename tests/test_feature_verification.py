import json

import numpy as np
import pytest

from bimer.feature_store import FeatureShard, FeatureStore
from bimer.feature_verification import (
    verify_feature_range,
    write_range_completion,
)
from bimer.schema import UtteranceRecord


def make_records(count):
    return [
        UtteranceRecord(
            dataset="emotiontalk",
            split="train",
            dialogue_id="d1",
            utterance_id=index,
            text=f"line {index}",
            emotion="neutral",
            language="zh",
            start_seconds=float(index),
            end_seconds=float(index + 1),
            video_path=f"{index}.mp4",
        )
        for index in range(count)
    ]


def feature_shard(records, *, text_width=768):
    rows = len(records)
    return FeatureShard(
        sample_ids=np.asarray([record.sample_id for record in records]),
        text=np.ones((rows, text_width), dtype=np.float32),
        audio=np.ones((rows, 1024), dtype=np.float32),
        vision=np.ones((rows, 512), dtype=np.float32),
        modality_mask=np.ones((rows, 3), dtype=np.bool_),
    )


def write_valid_range(store, records, shard_size, offset):
    for local_index, start in enumerate(range(0, len(records), shard_size)):
        store.write(
            "emotiontalk",
            "train",
            offset + local_index,
            feature_shard(records[start : start + shard_size]),
        )


def test_verify_complete_feature_range(tmp_path):
    records = make_records(33)
    store = FeatureStore(tmp_path)
    write_valid_range(store, records, 16, 0)

    result = verify_feature_range(records, store, shard_size=16)

    assert result.sample_count == 33
    assert result.verified_shards == 3
    assert result.start_shard == 0
    assert result.end_shard == 3
    assert result.is_valid is True


def test_verify_partial_range_allows_other_global_shards(tmp_path):
    records = make_records(40)
    store = FeatureStore(tmp_path)
    write_valid_range(store, records[:16], 16, 0)
    write_valid_range(store, records[16:], 16, 1)

    result = verify_feature_range(
        records[16:],
        store,
        shard_size=16,
        shard_index_offset=1,
        total_shards=3,
    )

    assert result.sample_count == 24
    assert result.verified_shards == 2
    assert (result.start_shard, result.end_shard) == (1, 3)
    assert result.total_shards == 3


def test_verify_feature_range_rejects_missing_shard(tmp_path):
    records = make_records(33)
    store = FeatureStore(tmp_path)
    write_valid_range(store, records[:16], 16, 0)
    write_valid_range(store, records[32:], 16, 2)

    with pytest.raises(ValueError, match="missing shard 1"):
        verify_feature_range(records, store, shard_size=16)


def test_verify_feature_range_rejects_wrong_ids(tmp_path):
    records = make_records(16)
    store = FeatureStore(tmp_path)
    store.write("emotiontalk", "train", 0, feature_shard(list(reversed(records))))

    with pytest.raises(ValueError, match="unexpected sample IDs"):
        verify_feature_range(records, store, shard_size=16)


def test_verify_feature_range_rejects_wrong_width(tmp_path):
    records = make_records(16)
    store = FeatureStore(tmp_path)
    store.write("emotiontalk", "train", 0, feature_shard(records, text_width=767))

    with pytest.raises(ValueError, match="width 768"):
        verify_feature_range(records, store, shard_size=16)


def test_verify_feature_range_rejects_non_finite_payload(tmp_path):
    records = make_records(2)
    path = FeatureStore(tmp_path).path("emotiontalk", "train", 0)
    path.parent.mkdir(parents=True)
    text = np.ones((2, 768), dtype=np.float32)
    text[0, 0] = np.nan
    np.savez_compressed(
        path,
        sample_ids=np.asarray([record.sample_id for record in records]),
        text=text,
        audio=np.ones((2, 1024), dtype=np.float32),
        vision=np.ones((2, 512), dtype=np.float32),
        modality_mask=np.ones((2, 3), dtype=np.bool_),
    )

    with pytest.raises(ValueError, match="finite"):
        verify_feature_range(records, FeatureStore(tmp_path), shard_size=16)


def test_write_range_completion_is_atomic(tmp_path):
    records = make_records(16)
    store = FeatureStore(tmp_path)
    write_valid_range(store, records, 16, 0)
    result = verify_feature_range(records, store, shard_size=16)

    path = write_range_completion(result, tmp_path)

    assert path == tmp_path / "ranges" / "range-00000-00001.json"
    assert json.loads(path.read_text(encoding="utf-8"))["is_valid"] is True
    assert not list(tmp_path.rglob("*.tmp"))

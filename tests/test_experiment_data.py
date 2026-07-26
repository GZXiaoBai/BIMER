import numpy as np

from bimer.experiment_data import build_dialogue_examples
from bimer.feature_store import FeatureShard
from bimer.schema import UtteranceRecord


def _records():
    return [
        UtteranceRecord(
            dataset="meld",
            split="train",
            dialogue_id="d1",
            utterance_id=index,
            text=f"line {index}",
            emotion="joy" if index % 2 else "neutral",
            language="en",
            start_seconds=float(index),
            end_seconds=float(index + 1),
        )
        for index in range(4)
    ]


def test_build_dialogue_examples_aligns_shards_and_overlapping_windows():
    records = _records()
    shard = FeatureShard(
        sample_ids=np.array([record.sample_id for record in reversed(records)]),
        text=np.arange(16, dtype=np.float32).reshape(4, 4),
        audio=np.arange(24, dtype=np.float32).reshape(4, 6),
        vision=np.arange(20, dtype=np.float32).reshape(4, 5),
        modality_mask=np.ones((4, 3), dtype=np.bool_),
    )
    examples = build_dialogue_examples(records, [shard], max_length=3, overlap=1)
    assert len(examples) == 2
    assert examples[0].sample_ids == tuple(record.sample_id for record in records[:3])
    assert examples[1].sample_ids == tuple(record.sample_id for record in records[2:])
    assert examples[0].language_id == 0
    assert examples[0].labels.tolist() == [0, 1, 0]
    assert examples[0].modality_quality.shape == (3, 3, 4)


def test_build_dialogue_examples_rejects_missing_cached_feature():
    records = _records()
    shard = FeatureShard(
        sample_ids=np.array([records[0].sample_id]),
        text=np.ones((1, 4), np.float32),
        audio=np.ones((1, 6), np.float32),
        vision=np.ones((1, 5), np.float32),
        modality_mask=np.ones((1, 3), np.bool_),
    )
    try:
        build_dialogue_examples(records, [shard])
    except ValueError as exc:
        assert "missing cached features" in str(exc)
    else:
        raise AssertionError("missing features were accepted")

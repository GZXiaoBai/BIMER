import numpy as np

from bimer.corruption_sampling import (
    materialize_feature_subset,
    select_stratified_context_records,
)
from bimer.feature_store import FeatureShard, FeatureStore
from bimer.schema import UtteranceRecord


def _record(dataset, context, utterance, emotion):
    return UtteranceRecord(
        dataset=dataset,
        split="train",
        dialogue_id=f"{context}-speaker",
        context_id=context,
        utterance_id=utterance,
        text="line",
        emotion=emotion,
        language="en" if dataset == "meld" else "zh",
        start_seconds=float(utterance),
        end_seconds=float(utterance + 1),
    )


def test_stratified_selection_keeps_whole_contexts_and_is_deterministic():
    records = []
    for dataset in ("meld", "emotiontalk"):
        for emotion in ("neutral", "joy"):
            for context_index in range(10):
                context = f"{dataset}-{emotion}-{context_index}"
                records.extend(
                    [_record(dataset, context, 0, emotion), _record(dataset, context, 1, emotion)]
                )

    first = select_stratified_context_records(records, fraction=0.1, seed=42)
    second = select_stratified_context_records(records, fraction=0.1, seed=42)

    assert [record.sample_id for record in first] == [record.sample_id for record in second]
    selected_contexts = {record.effective_context_id for record in first}
    assert len(selected_contexts) == 4
    assert len(first) == 8
    for context in selected_contexts:
        assert sum(record.effective_context_id == context for record in first) == 2


def test_materialize_feature_subset_reorders_by_manifest_and_preserves_quality(tmp_path):
    records = [_record("meld", "d", index, "neutral") for index in range(3)]
    base = FeatureStore(tmp_path / "base")
    quality = np.arange(3 * 3 * 4, dtype=np.float32).reshape(3, 3, 4) / 100
    base.write(
        "meld",
        "train",
        0,
        FeatureShard(
            sample_ids=np.asarray([record.sample_id for record in records]),
            text=np.arange(12, dtype=np.float32).reshape(3, 4),
            audio=np.arange(18, dtype=np.float32).reshape(3, 6),
            vision=np.arange(15, dtype=np.float32).reshape(3, 5),
            modality_mask=np.ones((3, 3), dtype=np.bool_),
            modality_quality=quality,
        ),
    )

    output = FeatureStore(tmp_path / "subset")
    paths = materialize_feature_subset(
        [records[2], records[0]], base, output, shard_size=1
    )

    assert len(paths) == 2
    merged = output.read_all("meld", "train")
    assert [str(shard.sample_ids[0]) for shard in merged] == [
        records[2].sample_id,
        records[0].sample_id,
    ]
    np.testing.assert_allclose(merged[0].modality_quality[0], quality[2])

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from .feature_store import FeatureShard
from .labels import emotion_index
from .schema import UtteranceRecord
from .training import DialogueExample
from .windowing import make_context_windows


def build_dialogue_examples(
    records: Iterable[UtteranceRecord],
    shards: Iterable[FeatureShard],
    *,
    max_length: int = 32,
    overlap: int = 8,
) -> list[DialogueExample]:
    feature_rows: dict[
        str,
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    ] = {}
    for shard in shards:
        for index, sample_id in enumerate(shard.sample_ids.tolist()):
            key = str(sample_id)
            if key in feature_rows:
                raise ValueError(f"duplicate cached features for {key}")
            feature_rows[key] = (
                shard.text[index],
                shard.audio[index],
                shard.vision[index],
                shard.modality_mask[index],
                np.asarray(shard.modality_quality)[index],
            )

    materialized = list(records)
    missing = sorted(
        record.sample_id for record in materialized if record.sample_id not in feature_rows
    )
    if missing:
        preview = ", ".join(missing[:3])
        raise ValueError(f"missing cached features for {len(missing)} records: {preview}")

    examples: list[DialogueExample] = []
    for window in make_context_windows(materialized, max_length=max_length, overlap=overlap):
        rows = [feature_rows[record.sample_id] for record in window.records]
        examples.append(
            DialogueExample(
                dataset=window.records[0].dataset,
                sample_ids=tuple(record.sample_id for record in window.records),
                text=np.stack([row[0] for row in rows]).astype(np.float32),
                audio=np.stack([row[1] for row in rows]).astype(np.float32),
                vision=np.stack([row[2] for row in rows]).astype(np.float32),
                modality_mask=np.stack([row[3] for row in rows]).astype(np.bool_),
                modality_quality=np.stack([row[4] for row in rows]).astype(np.float32),
                labels=np.asarray(
                    [emotion_index(str(record.emotion)) for record in window.records],
                    dtype=np.int64,
                ),
                language_id=0 if window.records[0].language == "en" else 1,
            )
        )
    return examples

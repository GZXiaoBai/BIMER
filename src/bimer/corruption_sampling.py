from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from .feature_store import FeatureShard, FeatureStore
from .labels import EMOTION_LABELS
from .schema import UtteranceRecord


def _stratum_seed(seed: int, dataset: str, emotion: str) -> int:
    digest = hashlib.sha256(
        f"{seed}\0{dataset}\0{emotion}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "little", signed=False)


def select_stratified_context_records(
    records: Iterable[UtteranceRecord],
    *,
    fraction: float = 0.1,
    seed: int = 42,
) -> list[UtteranceRecord]:
    """Select whole training conversations, stratified by dataset/dominant label."""
    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must be within (0, 1]")
    materialized = list(records)
    if any(str(record.split) != "train" for record in materialized):
        raise ValueError("corruption sampling accepts training records only")
    contexts: dict[tuple[str, str], list[UtteranceRecord]] = defaultdict(list)
    for record in materialized:
        contexts[(record.dataset, record.effective_context_id)].append(record)

    label_order = {label: index for index, label in enumerate(EMOTION_LABELS)}
    strata: dict[tuple[str, str], list[str]] = defaultdict(list)
    for (dataset, context_id), group in contexts.items():
        counts = Counter(str(record.emotion) for record in group)
        dominant = max(
            counts,
            key=lambda label: (counts[label], -label_order.get(label, 999)),
        )
        strata[(dataset, dominant)].append(context_id)

    selected: set[tuple[str, str]] = set()
    for (dataset, emotion), context_ids in sorted(strata.items()):
        ordered = sorted(context_ids)
        count = max(1, int(round(len(ordered) * fraction)))
        generator = np.random.default_rng(_stratum_seed(seed, dataset, emotion))
        positions = generator.choice(len(ordered), size=count, replace=False)
        selected.update((dataset, ordered[int(position)]) for position in positions)
    return [
        record
        for record in materialized
        if (record.dataset, record.effective_context_id) in selected
    ]


def materialize_feature_subset(
    records: Sequence[UtteranceRecord],
    base_store: FeatureStore,
    output_store: FeatureStore,
    *,
    shard_size: int = 1024,
) -> list[Path]:
    """Write a compact feature root in manifest order without recomputing encoders."""
    if shard_size <= 0:
        raise ValueError("shard_size must be positive")
    written: list[Path] = []
    groups = sorted({(record.dataset, str(record.split)) for record in records})
    for dataset, split in groups:
        group = [
            record
            for record in records
            if record.dataset == dataset and str(record.split) == split
        ]
        rows: dict[str, tuple[np.ndarray, ...]] = {}
        for shard in base_store.read_all(dataset, split):
            for index, sample_id in enumerate(shard.sample_ids.astype(str)):
                rows[str(sample_id)] = (
                    shard.text[index],
                    shard.audio[index],
                    shard.vision[index],
                    shard.modality_mask[index],
                    np.asarray(shard.modality_quality)[index],
                )
        missing = [record.sample_id for record in group if record.sample_id not in rows]
        if missing:
            raise ValueError(
                f"base features missing {len(missing)} selected samples: {missing[0]}"
            )
        for shard_index, start in enumerate(range(0, len(group), shard_size)):
            chunk = group[start : start + shard_size]
            sample_ids = np.asarray([record.sample_id for record in chunk], dtype=str)
            path = output_store.path(dataset, split, shard_index)
            if path.is_file():
                existing = output_store.read(path)
                if not np.array_equal(existing.sample_ids.astype(str), sample_ids):
                    raise ValueError(f"existing subset shard {path} has unexpected IDs")
                written.append(path)
                continue
            selected = [rows[record.sample_id] for record in chunk]
            written.append(
                output_store.write(
                    dataset,
                    split,
                    shard_index,
                    FeatureShard(
                        sample_ids=sample_ids,
                        text=np.stack([row[0] for row in selected]),
                        audio=np.stack([row[1] for row in selected]),
                        vision=np.stack([row[2] for row in selected]),
                        modality_mask=np.stack([row[3] for row in selected]),
                        modality_quality=np.stack([row[4] for row in selected]),
                    ),
                )
            )
    return written

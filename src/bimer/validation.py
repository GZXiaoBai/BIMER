from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Iterable

from .schema import UtteranceRecord


@dataclass(frozen=True, slots=True)
class DatasetValidationReport:
    split_counts: dict[str, int]
    label_counts: dict[str, int]
    duplicate_sample_ids: tuple[str, ...]
    cross_split_media: tuple[str, ...]

    @property
    def is_valid(self) -> bool:
        return not self.duplicate_sample_ids and not self.cross_split_media


def validate_dataset_records(
    records: Iterable[UtteranceRecord],
) -> DatasetValidationReport:
    materialized = list(records)
    sample_counts = Counter(record.sample_id for record in materialized)
    split_counts = Counter(str(record.split) for record in materialized)
    label_counts = Counter(str(record.emotion) for record in materialized)

    media_splits: dict[str, set[str]] = defaultdict(set)
    for record in materialized:
        for path in (record.video_path, record.audio_path):
            if path is not None:
                media_splits[str(path)].add(str(record.split))

    return DatasetValidationReport(
        split_counts=dict(split_counts),
        label_counts=dict(label_counts),
        duplicate_sample_ids=tuple(
            sorted(sample_id for sample_id, count in sample_counts.items() if count > 1)
        ),
        cross_split_media=tuple(
            sorted(path for path, splits in media_splits.items() if len(splits) > 1)
        ),
    )

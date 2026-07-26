from __future__ import annotations

from concurrent.futures import Executor, ProcessPoolExecutor
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from .feature_store import FeatureShard, FeatureStore
from .parallel_feature_extraction import prefetched_map
from .quality import text_quality
from .schema import UtteranceRecord


class QualityAttachmentRunner:
    """Measure raw-media quality while reusing every cached encoder feature."""

    def __init__(
        self,
        *,
        audio_quality_loader: Callable[[Path], np.ndarray],
        vision_quality_loader: Callable[[Path], np.ndarray],
        audio_executor_factory: Callable[..., Executor] = ProcessPoolExecutor,
        vision_executor_factory: Callable[..., Executor] = ProcessPoolExecutor,
        workers: int = 4,
        queue_capacity: int = 8,
    ) -> None:
        self.audio_quality_loader = audio_quality_loader
        self.vision_quality_loader = vision_quality_loader
        self.audio_executor_factory = audio_executor_factory
        self.vision_executor_factory = vision_executor_factory
        self.workers = workers
        self.queue_capacity = queue_capacity

    def run(
        self,
        records: Sequence[UtteranceRecord],
        base_store: FeatureStore,
        output_store: FeatureStore,
        *,
        start_shard: int | None = None,
        end_shard: int | None = None,
    ) -> list[Path]:
        if not records:
            return []
        groups = {(record.dataset, str(record.split)) for record in records}
        if len(groups) != 1:
            raise ValueError("attach quality to one dataset split at a time")
        dataset, split = next(iter(groups))
        record_by_id = {record.sample_id: record for record in records}
        if len(record_by_id) != len(records):
            raise ValueError("quality manifest contains duplicate sample IDs")
        selected: list[tuple[int, FeatureShard]] = []
        written: list[Path] = []
        for path in base_store.paths(dataset, split):
            shard_index = int(path.stem.rsplit("-", 1)[1])
            if start_shard is not None and shard_index < start_shard:
                continue
            if end_shard is not None and shard_index >= end_shard:
                continue
            base = base_store.read(path)
            output_path = output_store.path(dataset, split, shard_index)
            if output_path.is_file():
                existing = output_store.read(output_path)
                if not np.array_equal(existing.sample_ids.astype(str), base.sample_ids.astype(str)):
                    raise ValueError(f"existing quality shard {output_path} has unexpected IDs")
                written.append(output_path)
                continue
            missing = [
                sample_id
                for sample_id in base.sample_ids.astype(str)
                if sample_id not in record_by_id
            ]
            if missing:
                raise ValueError(f"manifest is missing {len(missing)} base samples: {missing[0]}")
            selected.append((shard_index, base))

        flattened_records = [
            record_by_id[sample_id]
            for _, shard in selected
            for sample_id in shard.sample_ids.astype(str)
        ]
        media_paths = [
            Path(record.video_path) if record.video_path is not None else Path("__missing_media__")
            for record in flattened_records
        ]
        audio_rows = list(
            prefetched_map(
                self.audio_quality_loader,
                media_paths,
                workers=self.workers,
                queue_capacity=self.queue_capacity,
                executor_factory=self.audio_executor_factory,
            )
        )
        vision_rows = list(
            prefetched_map(
                self.vision_quality_loader,
                media_paths,
                workers=self.workers,
                queue_capacity=self.queue_capacity,
                executor_factory=self.vision_executor_factory,
            )
        )
        offset = 0
        for shard_index, base in selected:
            rows = len(base.sample_ids)
            chunk_records = flattened_records[offset : offset + rows]
            quality = np.stack(
                (
                    np.stack(
                        [
                            text_quality(
                                record.text,
                                source=record.text_source,
                                asr_confidence=record.asr_confidence,
                            )
                            for record in chunk_records
                        ]
                    ),
                    np.stack(audio_rows[offset : offset + rows]),
                    np.stack(vision_rows[offset : offset + rows]),
                ),
                axis=1,
            ).astype(np.float32)
            mask = np.asarray(base.modality_mask, dtype=np.bool_)
            quality[~mask] = 0.0
            written.append(
                output_store.write(
                    dataset,
                    split,
                    shard_index,
                    FeatureShard(
                        sample_ids=base.sample_ids,
                        text=base.text,
                        audio=base.audio,
                        vision=base.vision,
                        modality_mask=base.modality_mask,
                        modality_quality=quality,
                    ),
                )
            )
            offset += rows
        return sorted(written)

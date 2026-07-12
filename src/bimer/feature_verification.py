from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Sequence

from .feature_store import FeatureStore
from .modality_store import verified_final_shard
from .parallel_feature_extraction import record_shards
from .schema import UtteranceRecord


@dataclass(frozen=True, slots=True)
class FeatureVerificationResult:
    dataset: str
    split: str
    sample_count: int
    expected_shards: int
    verified_shards: int
    start_shard: int
    end_shard: int
    total_shards: int
    is_valid: bool = True

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _shard_index(path: Path) -> int:
    prefix = "features-"
    if not path.stem.startswith(prefix):
        raise ValueError(f"unexpected feature filename: {path.name}")
    try:
        return int(path.stem[len(prefix) :])
    except ValueError as error:
        raise ValueError(f"unexpected feature filename: {path.name}") from error


def verify_feature_range(
    records: Sequence[UtteranceRecord],
    store: FeatureStore,
    *,
    shard_size: int,
    shard_index_offset: int = 0,
    total_shards: int | None = None,
) -> FeatureVerificationResult:
    if not records:
        raise ValueError("feature verification requires at least one record")
    groups = {(record.dataset, str(record.split)) for record in records}
    if len(groups) != 1:
        raise ValueError("verify one dataset split at a time")
    dataset, split = next(iter(groups))
    record_ids = [record.sample_id for record in records]
    if len(set(record_ids)) != len(record_ids):
        raise ValueError("manifest sample IDs must be unique")

    expected = list(record_shards(records, shard_size, shard_index_offset))
    start_shard = shard_index_offset
    end_shard = start_shard + len(expected)
    resolved_total = end_shard if total_shards is None else total_shards
    if resolved_total < end_shard:
        raise ValueError("total_shards cannot be smaller than end_shard")

    expected_indices = {index for index, _, _ in expected}
    if start_shard == 0 and end_shard == resolved_total:
        actual_indices = {
            _shard_index(path) for path in store.paths(dataset, split)
        }
        extras = sorted(actual_indices - expected_indices)
        if extras:
            raise ValueError(f"unexpected feature shards: {extras}")

    verified_ids: list[str] = []
    for shard_index, _, expected_ids in expected:
        path = verified_final_shard(
            store,
            dataset,
            split,
            shard_index,
            expected_ids,
        )
        if path is None:
            raise ValueError(f"missing shard {shard_index}")
        shard = store.read(path)
        verified_ids.extend(shard.sample_ids.astype(str).tolist())

    if len(set(verified_ids)) != len(verified_ids):
        raise ValueError("verified sample IDs must be unique")
    if verified_ids != record_ids:
        raise ValueError("verified sample IDs do not match manifest order")

    return FeatureVerificationResult(
        dataset=dataset,
        split=split,
        sample_count=len(records),
        expected_shards=len(expected),
        verified_shards=len(expected),
        start_shard=start_shard,
        end_shard=end_shard,
        total_shards=resolved_total,
    )


def write_range_completion(
    result: FeatureVerificationResult,
    feature_root: Path | str,
) -> Path:
    if not result.is_valid:
        raise ValueError("cannot publish an invalid verification result")
    path = (
        Path(feature_root)
        / "ranges"
        / f"range-{result.start_shard:05d}-{result.end_shard:05d}.json"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        temporary.write_text(
            json.dumps(result.to_dict(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return path

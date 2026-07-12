from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, TypeVar


RecordT = TypeVar("RecordT")


@dataclass(frozen=True, slots=True)
class ShardRange:
    start: int
    end: int
    total_shards: int

    @property
    def shard_count(self) -> int:
        return self.end - self.start


def resolve_shard_range(
    record_count: int,
    shard_size: int,
    start_shard: int | None,
    end_shard: int | None,
) -> ShardRange:
    if record_count < 0:
        raise ValueError("record_count must be non-negative")
    if shard_size <= 0:
        raise ValueError("shard_size must be positive")
    total_shards = (record_count + shard_size - 1) // shard_size
    if start_shard is None and end_shard is None:
        return ShardRange(0, total_shards, total_shards)
    if start_shard is None or end_shard is None:
        raise ValueError(
            "start-shard and end-shard must be supplied together"
        )
    if not 0 <= start_shard < end_shard <= total_shards:
        raise ValueError(
            "shard range must satisfy "
            f"0 <= start < end <= {total_shards}"
        )
    return ShardRange(start_shard, end_shard, total_shards)


def slice_shard_range(
    records: Sequence[RecordT],
    shard_size: int,
    start_shard: int | None,
    end_shard: int | None,
) -> tuple[list[RecordT], ShardRange]:
    resolved = resolve_shard_range(
        len(records), shard_size, start_shard, end_shard
    )
    start = resolved.start * shard_size
    end = min(resolved.end * shard_size, len(records))
    return list(records[start:end]), resolved

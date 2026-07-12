import pytest

from bimer.shard_ranges import ShardRange, resolve_shard_range, slice_shard_range


def test_resolve_full_and_partial_shard_ranges():
    assert resolve_shard_range(15413, 16, None, None) == ShardRange(
        0, 964, 964
    )
    assert resolve_shard_range(15413, 16, 120, 240) == ShardRange(
        120, 240, 964
    )


@pytest.mark.parametrize(
    ("start", "end"),
    [(0, None), (None, 1), (-1, 1), (2, 2), (3, 2), (0, 965)],
)
def test_resolve_shard_range_rejects_invalid_bounds(start, end):
    with pytest.raises(ValueError):
        resolve_shard_range(15413, 16, start, end)


def test_slice_shard_range_keeps_short_final_shard():
    records = list(range(15413))

    selected, resolved = slice_shard_range(records, 16, 960, 964)

    assert resolved == ShardRange(960, 964, 964)
    assert selected[0] == 15360
    assert selected[-1] == 15412


def test_resolve_empty_full_range():
    assert resolve_shard_range(0, 16, None, None) == ShardRange(0, 0, 0)

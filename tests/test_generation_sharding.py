from __future__ import annotations

import pytest

from scripts.generate import _select_shard


def test_two_shards_are_balanced_disjoint_and_complete() -> None:
    items = list(range(401))

    left = _select_shard(items, shard_count=2, shard_index=0)
    right = _select_shard(items, shard_count=2, shard_index=1)

    assert abs(len(left) - len(right)) == 1
    assert set(left).isdisjoint(right)
    assert sorted([*left, *right]) == items
    assert left == items[::2]
    assert right == items[1::2]


@pytest.mark.parametrize(
    ("shard_count", "shard_index"),
    [(0, 0), (2, -1), (2, 2)],
)
def test_invalid_shard_parameters_are_rejected(
    shard_count: int,
    shard_index: int,
) -> None:
    with pytest.raises(ValueError):
        _select_shard([1, 2], shard_count=shard_count, shard_index=shard_index)

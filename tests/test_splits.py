"""Anti-leakage tests -- written BEFORE any model, as per the project rules.

These tests fail if train/test indices overlap, if temporal order is violated, if the
purge+embargo gap is not respected, or if anything resembling a shuffle sneaks in.
"""

from __future__ import annotations

import numpy as np
import pytest

from engine.splits import LeakageError, assert_no_leakage, temporal_train_test_split


def test_split_is_disjoint_and_temporally_ordered() -> None:
    split = temporal_train_test_split(1000, test_size=0.3, purge=5, embargo=10)
    assert np.intersect1d(split.train, split.test).size == 0
    assert split.train.max() < split.test.min()


def test_gap_equals_purge_plus_embargo() -> None:
    split = temporal_train_test_split(1000, test_size=0.3, purge=5, embargo=10)
    # n_test = 300 -> test starts at 700; train must end 15 bars earlier
    assert split.test.min() == 700
    assert split.train.max() == 684
    gap = int(split.test.min() - split.train.max() - 1)
    assert gap == 15


def test_no_shuffle_indices_strictly_increasing() -> None:
    split = temporal_train_test_split(500, test_size=0.2, purge=3, embargo=2)
    assert (np.diff(split.train) > 0).all()
    assert (np.diff(split.test) > 0).all()


def test_full_coverage_when_no_purge_no_embargo() -> None:
    split = temporal_train_test_split(100, test_size=0.25)
    combined = np.concatenate([split.train, split.test])
    assert np.array_equal(np.sort(combined), np.arange(100))


def test_validator_catches_overlap() -> None:
    with pytest.raises(LeakageError, match="overlap"):
        assert_no_leakage(np.arange(0, 100), np.arange(90, 150))


def test_validator_catches_shuffle() -> None:
    shuffled = np.array([5, 2, 8, 1, 3])
    with pytest.raises(LeakageError, match="shuffl"):
        assert_no_leakage(shuffled, np.arange(100, 150))


def test_validator_catches_wrong_temporal_order() -> None:
    with pytest.raises(LeakageError, match="temporal order"):
        assert_no_leakage(np.arange(100, 200), np.arange(0, 50))


def test_validator_catches_insufficient_embargo_gap() -> None:
    with pytest.raises(LeakageError, match="gap"):
        assert_no_leakage(np.arange(0, 100), np.arange(102, 150), min_gap=5)


def test_validator_accepts_clean_split() -> None:
    assert_no_leakage(np.arange(0, 100), np.arange(110, 150), min_gap=10)


@pytest.mark.parametrize("test_size", [0.0, 1.0, -0.1, 1.5])
def test_invalid_test_size_raises(test_size: float) -> None:
    with pytest.raises(ValueError):
        temporal_train_test_split(100, test_size=test_size)


def test_negative_purge_or_embargo_raises() -> None:
    with pytest.raises(ValueError):
        temporal_train_test_split(100, purge=-1)
    with pytest.raises(ValueError):
        temporal_train_test_split(100, embargo=-1)


def test_purge_larger_than_train_raises() -> None:
    with pytest.raises(ValueError, match="empty"):
        temporal_train_test_split(100, test_size=0.5, purge=40, embargo=20)

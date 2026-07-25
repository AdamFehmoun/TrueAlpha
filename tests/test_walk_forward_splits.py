"""Walk-forward K-fold splits: coverage, gaps, anchoring, explicit failures.

Every geometric property is asserted fold by fold, and hand-tampered folds must be
REJECTED by ``assert_no_leakage`` -- the same validator the evaluation pipeline calls.
"""

from __future__ import annotations

import numpy as np
import pytest

from engine.splits import (
    LeakageError,
    TemporalSplit,
    assert_no_leakage,
    temporal_train_test_split,
    walk_forward_splits,
)


def test_returns_exactly_n_folds() -> None:
    splits = walk_forward_splits(1000, n_folds=5, test_size=0.3, purge=10, embargo=5)
    assert len(splits) == 5


def test_test_windows_are_disjoint_and_strictly_increasing() -> None:
    splits = walk_forward_splits(1000, n_folds=5, test_size=0.3, purge=10, embargo=5)
    all_test = np.concatenate([s.test for s in splits])
    # strictly increasing across the concatenation = disjoint AND chronological
    assert (np.diff(all_test) > 0).all()


def test_union_of_test_windows_is_exactly_the_tail() -> None:
    splits = walk_forward_splits(1000, n_folds=5, test_size=0.3)
    union = np.concatenate([s.test for s in splits])
    assert np.array_equal(union, np.arange(700, 1000))


@pytest.mark.parametrize(
    ("n_samples", "n_folds", "test_size"),
    [(1096, 5, 0.2), (997, 7, 0.31), (100, 3, 0.25)],
)
def test_indivisible_test_region_still_tiles_the_tail_with_balanced_windows(
    n_samples: int, n_folds: int, test_size: float
) -> None:
    splits = walk_forward_splits(n_samples, n_folds=n_folds, test_size=test_size)
    n_test = int(round(n_samples * test_size))
    union = np.concatenate([s.test for s in splits])
    assert np.array_equal(union, np.arange(n_samples - n_test, n_samples))
    sizes = [len(s.test) for s in splits]
    assert max(sizes) - min(sizes) <= 1
    # (i*n)//k boundaries spread extra bars evenly; fold 0 is never the one that
    # gets an extra bar and the last fold always does
    assert len(splits[0].test) == min(sizes)
    assert len(splits[-1].test) == max(sizes)


def test_gap_equals_purge_plus_embargo_on_every_fold() -> None:
    for anchored in (True, False):
        splits = walk_forward_splits(
            1000, n_folds=4, test_size=0.4, purge=7, embargo=3, anchored=anchored
        )
        for s in splits:
            assert int(s.test.min()) - int(s.train.max()) - 1 == 10


def test_anchored_train_always_starts_at_zero_and_strictly_grows() -> None:
    splits = walk_forward_splits(1000, n_folds=5, test_size=0.3, purge=5, embargo=5, anchored=True)
    lengths = [len(s.train) for s in splits]
    assert all(int(s.train[0]) == 0 for s in splits)
    assert all(b > a for a, b in zip(lengths, lengths[1:], strict=False))


def test_rolling_train_has_constant_length_and_slides_forward() -> None:
    rolling = walk_forward_splits(
        1000, n_folds=5, test_size=0.3, purge=5, embargo=5, anchored=False
    )
    assert len({len(s.train) for s in rolling}) == 1
    starts = [int(s.train[0]) for s in rolling]
    assert all(b > a for a, b in zip(starts, starts[1:], strict=False))
    # rolling and anchored agree on the FIRST fold by construction
    anchored = walk_forward_splits(1000, n_folds=5, test_size=0.3, purge=5, embargo=5)
    assert np.array_equal(rolling[0].train, anchored[0].train)


def test_every_returned_fold_passes_the_leakage_validator() -> None:
    for anchored in (True, False):
        splits = walk_forward_splits(
            500, n_folds=4, test_size=0.25, purge=8, embargo=2, anchored=anchored
        )
        for s in splits:
            assert_no_leakage(s.train, s.test, min_gap=s.purge + s.embargo)


def test_single_fold_matches_the_single_forward_split_geometry() -> None:
    (wf,) = walk_forward_splits(1000, n_folds=1, test_size=0.3, purge=5, embargo=10)
    single = temporal_train_test_split(1000, test_size=0.3, purge=5, embargo=10)
    assert np.array_equal(wf.train, single.train)
    assert np.array_equal(wf.test, single.test)


def test_the_m2_protocol_configuration_fits_1096_daily_bars() -> None:
    """The exact configuration of the walk-forward evaluation: 1096 bars, 5 folds,
    test_size 0.2, purge 200 (= slow_max), embargo 5. Pins the bar budget explicitly
    instead of discovering it at evaluation time: 219 test bars from position 877,
    672 train bars on fold 0."""
    splits = walk_forward_splits(1096, n_folds=5, test_size=0.2, purge=200, embargo=5)
    assert len(splits) == 5
    assert splits[0].train.size == 672
    union = np.concatenate([s.test for s in splits])
    assert np.array_equal(union, np.arange(877, 1096))


def test_too_many_folds_raises_explicitly() -> None:
    with pytest.raises(ValueError, match="non-empty folds"):
        walk_forward_splits(100, n_folds=30, test_size=0.2)


def test_empty_train_raises_explicitly_instead_of_shrinking_purge() -> None:
    # the test region starts at bar 70; purge+embargo = 70 consume the whole train
    with pytest.raises(ValueError, match="empty train"):
        walk_forward_splits(100, n_folds=3, test_size=0.3, purge=60, embargo=10)


def test_invalid_parameters_raise() -> None:
    with pytest.raises(ValueError, match="n_samples"):
        walk_forward_splits(0, n_folds=2, test_size=0.2)
    with pytest.raises(ValueError, match="n_samples"):
        walk_forward_splits(-100, n_folds=2, test_size=0.2)
    with pytest.raises(ValueError, match="n_folds"):
        walk_forward_splits(100, n_folds=0, test_size=0.2)
    with pytest.raises(ValueError, match="test_size"):
        walk_forward_splits(100, n_folds=2, test_size=1.5)
    with pytest.raises(ValueError, match="purge"):
        walk_forward_splits(100, n_folds=2, test_size=0.2, purge=-1)
    with pytest.raises(ValueError, match="purge"):
        walk_forward_splits(100, n_folds=2, test_size=0.2, embargo=-1)


def test_tampered_fold_with_train_test_overlap_is_rejected() -> None:
    s = walk_forward_splits(300, n_folds=3, test_size=0.3, purge=5, embargo=5)[0]
    tampered = TemporalSplit(
        train=np.arange(0, int(s.test.min()) + 1, dtype=np.int64),  # reaches into test
        test=s.test,
        purge=s.purge,
        embargo=s.embargo,
    )
    with pytest.raises(LeakageError):
        assert_no_leakage(tampered.train, tampered.test, min_gap=s.purge + s.embargo)


def test_tampered_fold_with_reversed_order_is_rejected() -> None:
    s = walk_forward_splits(300, n_folds=3, test_size=0.3)[1]
    with pytest.raises(LeakageError, match="temporal order"):
        assert_no_leakage(s.test, s.train, min_gap=0)  # train placed after test


def test_tampered_fold_with_insufficient_gap_is_rejected() -> None:
    s = walk_forward_splits(300, n_folds=3, test_size=0.3, purge=10, embargo=5)[0]
    stretched = np.arange(0, int(s.train.max()) + 6, dtype=np.int64)  # eats 5 gap bars
    with pytest.raises(LeakageError, match="gap"):
        assert_no_leakage(stretched, s.test, min_gap=s.purge + s.embargo)


def test_tampered_fold_with_shuffled_train_is_rejected() -> None:
    s = walk_forward_splits(300, n_folds=3, test_size=0.3)[0]
    shuffled = np.random.default_rng(0).permutation(s.train)
    with pytest.raises(LeakageError, match="shuffl"):
        assert_no_leakage(shuffled, s.test, min_gap=0)

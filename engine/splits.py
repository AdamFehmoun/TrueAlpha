"""Temporal train/test splits with purge and embargo. No shuffling, ever.

Written BEFORE any model exists, on purpose: any future strategy evaluation must go
through these splits, and any custom split must pass ``assert_no_leakage`` before its
backtest results are trusted.

Two constructors, same conventions (train strictly before test in every split):
- ``temporal_train_test_split``: one forward split.
- ``walk_forward_splits``: K chronological folds whose test windows tile the tail of
  the sample (anchored/expanding or rolling train).
- ``purge``: bars dropped from the end of train whose labels/features could overlap the
  test window (label horizon).
- ``embargo``: additional buffer bars dropped between train and test to break serial
  correlation leakage.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

IndexArray = npt.NDArray[np.int64]


class LeakageError(RuntimeError):
    """Raised when a train/test split could leak information."""


@dataclass(frozen=True)
class TemporalSplit:
    train: IndexArray
    test: IndexArray
    purge: int
    embargo: int


def temporal_train_test_split(
    n_samples: int,
    test_size: float = 0.3,
    purge: int = 0,
    embargo: int = 0,
) -> TemporalSplit:
    """Split positions [0, n_samples) into train, then a purge+embargo gap, then test."""
    if not 0.0 < test_size < 1.0:
        raise ValueError(f"test_size must be in (0, 1), got {test_size}")
    if purge < 0 or embargo < 0:
        raise ValueError("purge and embargo must be >= 0")

    n_test = int(round(n_samples * test_size))
    if n_test < 1:
        raise ValueError("test set is empty; increase test_size or n_samples")
    test_start = n_samples - n_test
    train_end = test_start - purge - embargo
    if train_end < 1:
        raise ValueError("train set is empty after purge/embargo")

    split = TemporalSplit(
        train=np.arange(0, train_end, dtype=np.int64),
        test=np.arange(test_start, n_samples, dtype=np.int64),
        purge=purge,
        embargo=embargo,
    )
    # A split constructed here must always pass its own guard.
    assert_no_leakage(split.train, split.test, min_gap=purge + embargo)
    return split


def walk_forward_splits(
    n_samples: int,
    n_folds: int,
    test_size: float,
    purge: int = 0,
    embargo: int = 0,
    anchored: bool = True,
) -> list[TemporalSplit]:
    """K chronological walk-forward folds over positions [0, n_samples).

    The last ``round(n_samples * test_size)`` bars form the out-of-sample region,
    cut into ``n_folds`` contiguous, disjoint, chronologically ordered test windows
    that exactly cover that tail. When the region is not divisible, fold boundaries
    sit at ``(i * n_test) // n_folds`` (integer arithmetic): window lengths differ by
    at most one bar, extra bars are spread as evenly as possible (fold 0 always gets
    the minimum length, the last fold the maximum, intermediate folds interleave).

    Each fold is ``train -> (purge + embargo) gap -> test``:

    - ``anchored=True``: expanding window -- train always starts at 0 and grows.
    - ``anchored=False``: rolling window -- train keeps the FIXED length of the first
      fold's train and slides forward.

    Every returned split passes ``assert_no_leakage(train, test, purge + embargo)``
    at construction. If the parameters cannot produce ``n_folds`` valid folds, an
    EXPLICIT ValueError is raised -- the fold count is never truncated and
    purge/embargo are never reduced silently.
    """
    if n_samples < 1:
        raise ValueError(f"n_samples must be >= 1, got {n_samples}")
    if n_folds < 1:
        raise ValueError(f"n_folds must be >= 1, got {n_folds}")
    if not 0.0 < test_size < 1.0:
        raise ValueError(f"test_size must be in (0, 1), got {test_size}")
    if purge < 0 or embargo < 0:
        raise ValueError("purge and embargo must be >= 0")

    n_test_total = int(round(n_samples * test_size))
    if n_test_total < n_folds:
        raise ValueError(
            f"cannot cut {n_test_total} test bars (test_size={test_size} of "
            f"n_samples={n_samples}) into {n_folds} non-empty folds; reduce n_folds or "
            "increase test_size -- purge/embargo are never reduced silently"
        )
    region_start = n_samples - n_test_total
    first_train_end = region_start - purge - embargo
    if first_train_end < 1:
        raise ValueError(
            f"fold 0 has an empty train: the test region starts at bar {region_start} "
            f"and purge+embargo consume {purge + embargo} bars; reduce test_size or "
            "add data -- purge/embargo are never reduced silently"
        )

    splits: list[TemporalSplit] = []
    for i in range(n_folds):
        test_start = region_start + (i * n_test_total) // n_folds
        test_end = region_start + ((i + 1) * n_test_total) // n_folds
        train_end = test_start - purge - embargo
        train_start = 0 if anchored else train_end - first_train_end
        split = TemporalSplit(
            train=np.arange(train_start, train_end, dtype=np.int64),
            test=np.arange(test_start, test_end, dtype=np.int64),
            purge=purge,
            embargo=embargo,
        )
        # every fold must pass its own guard at construction, like the single split
        assert_no_leakage(split.train, split.test, min_gap=purge + embargo)
        splits.append(split)
    return splits


def assert_no_leakage(train: IndexArray, test: IndexArray, min_gap: int = 0) -> None:
    """Raise LeakageError unless train/test are disjoint, ordered, unshuffled and gapped.

    Use this on ANY split (including hand-built ones) before trusting a backtest.
    """
    if len(train) == 0 or len(test) == 0:
        raise LeakageError("train or test is empty")
    if np.intersect1d(train, test).size > 0:
        raise LeakageError("train and test indices overlap")
    if not (np.all(np.diff(train) > 0) and np.all(np.diff(test) > 0)):
        raise LeakageError("indices are not strictly increasing -- shuffling detected")
    if int(train.max()) >= int(test.min()):
        raise LeakageError("train extends past the start of test -- temporal order violated")
    gap = int(test.min()) - int(train.max()) - 1
    if gap < min_gap:
        raise LeakageError(f"gap between train and test is {gap}, need >= {min_gap}")

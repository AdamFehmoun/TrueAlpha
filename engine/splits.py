"""Temporal train/test split with purge and embargo. No shuffling, ever.

Written BEFORE any model exists, on purpose: any future strategy evaluation must go
through these splits, and any custom split must pass ``assert_no_leakage`` before its
backtest results are trusted.

Conventions (single forward split, train strictly before test):
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

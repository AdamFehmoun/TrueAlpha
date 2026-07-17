"""Deterministic synthetic price builders for tests."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


def make_prices(open_: Sequence[float], close: Sequence[float] | None = None) -> pd.DataFrame:
    """Build an OHLCV frame from explicit open (and optionally close) prices."""
    open_arr = np.asarray(open_, dtype="float64")
    close_arr = open_arr.copy() if close is None else np.asarray(close, dtype="float64")
    if len(open_arr) != len(close_arr):
        raise ValueError("open and close must have the same length")
    index = pd.date_range("2023-01-01", periods=len(open_arr), freq="D", tz="UTC")
    return pd.DataFrame(
        {
            "open": open_arr,
            "high": np.maximum(open_arr, close_arr),
            "low": np.minimum(open_arr, close_arr),
            "close": close_arr,
            "volume": np.ones(len(open_arr)),
        },
        index=index,
    )


def random_walk_prices(n: int, seed: int) -> pd.DataFrame:
    """Seeded random-walk prices with opens that gap away from the previous close."""
    rng = np.random.default_rng(seed)
    close = 100.0 * np.cumprod(1.0 + rng.normal(0.0005, 0.03, size=n))
    open_ = np.empty(n)
    open_[0] = 100.0
    open_[1:] = close[:-1] * (1.0 + rng.normal(0.0, 0.005, size=n - 1))
    return make_prices(open_.tolist(), close.tolist())

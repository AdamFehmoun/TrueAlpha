"""Moving-average crossover (long/flat, no shorting).

Long 1.0 when the fast MA of closes is strictly above the slow MA, flat otherwise.
During the slow-MA warmup the signal is 0.0 (flat) -- never NaN, never dropped bars,
so the backtest index always matches the price index.
"""

from __future__ import annotations

import pandas as pd


def ma_crossover_signal(prices: pd.DataFrame, fast: int = 20, slow: int = 50) -> pd.Series:
    if fast >= slow:
        raise ValueError(f"fast ({fast}) must be < slow ({slow})")
    close = prices["close"]
    fast_ma = close.rolling(fast).mean()
    slow_ma = close.rolling(slow).mean()
    signal = (fast_ma > slow_ma).astype("float64")
    signal.loc[slow_ma.isna()] = 0.0
    return signal.rename("signal")

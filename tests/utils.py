"""Deterministic synthetic price builders for tests."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
import pandas as pd

FloatArray = Sequence[float] | npt.NDArray[np.float64]


def make_prices(
    open_: FloatArray, close: FloatArray | None = None, freq: str = "D"
) -> pd.DataFrame:
    """Build an OHLCV frame from explicit open (and optionally close) prices.

    ``freq`` only sets the synthetic index spacing (use "h" for very long series that
    would overflow pandas' datetime64[ns] bounds with daily bars); the engine and the
    metrics never read the index frequency.
    """
    open_arr = np.asarray(open_, dtype="float64")
    close_arr = open_arr.copy() if close is None else np.asarray(close, dtype="float64")
    if len(open_arr) != len(close_arr):
        raise ValueError("open and close must have the same length")
    index = pd.date_range("2023-01-01", periods=len(open_arr), freq=freq, tz="UTC")
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


def regime_shift_prices(seed: int, n_zigzag: int = 120, n_cycles: int = 280) -> pd.DataFrame:
    """Two concatenated regimes whose OPTIMAL MA-crossover pair provably moves.

    Why the optimum moves (the construction is structural, not statistical):

    - Regime 1 (bars 0..n_zigzag): an upward drift (+0.1%/bar) carrying a PERIOD-2
      ZIGZAG of amplitude 5%. A fast crossover (e.g. 3/10) chases the zigzag: it
      crosses every bar or two, pays fees on each flip and systematically buys the
      zigzag tops -- deeply negative Sharpe. A slow pair (e.g. 10/40) averages the
      zigzag to ~zero and rides the drift. The slow pair wins by several Sharpe
      units.
    - Regime 2 (the rest): clean sine cycles of PERIOD 16 (amplitude 20%), far
      below a slow pair's look-back -- a 40-bar mean IS the cycle average, so the
      slow pair is systematically mistimed and loses, while the fast pair tracks
      the phase and captures the up-legs. The fast pair wins by several Sharpe
      units.

    A rolling train window that slides from regime 1 into regime 2 therefore FLIPS
    its argmax between {slow pair} and {fast pair}. The seeded noise (0.2%/bar)
    breaks exact symmetry without ever flipping the ranking: the margins above are
    seed-robust (verified over multiple seeds during calibration).
    """
    rng = np.random.default_rng(seed)
    t1 = np.arange(n_zigzag)
    zigzag = 100.0 * np.exp(0.001 * t1) * (1.0 + 0.05 * ((-1.0) ** t1))
    zigzag *= np.exp(np.cumsum(rng.normal(0.0, 0.002, n_zigzag)))
    t2 = np.arange(n_cycles)
    cycles = zigzag[-1] * (1.0 + 0.20 * np.sin(2.0 * np.pi * t2 / 16.0))
    cycles *= np.exp(np.cumsum(rng.normal(0.0, 0.002, n_cycles)))
    close = np.concatenate([zigzag, cycles])
    open_ = np.empty_like(close)
    open_[0] = close[0]
    open_[1:] = close[:-1]
    return make_prices(open_.tolist(), close.tolist())

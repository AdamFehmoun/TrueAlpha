"""Frequency-safe annualization: default pins and 1d-vs-1h coherence.

The default-pin tests exist because a mutation audit showed that sabotaging the daily
annualization constant (365 -> 8760) was only caught indirectly, by the vectorbt
oracle. These tests pin the DEFAULT behavior of every annualizing function directly,
with expectations computed from the literal constants -- so that mutation now fails
here, by name.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from engine.metrics import (
    BARS_PER_YEAR,
    CALENDAR_BARS_PER_YEAR,
    annualized_turnover,
    cagr,
    sharpe_ratio,
    sharpe_tstat,
    sortino_ratio,
)

# Fixed arbitrary daily returns; nothing special about the values.
_DAILY = pd.Series([0.010, -0.020, 0.030, 0.004, -0.012, 0.007, 0.015, -0.003])


def test_frequency_table_is_the_single_source_of_truth() -> None:
    assert BARS_PER_YEAR == {"1d": 365, "1h": 8760}
    assert CALENDAR_BARS_PER_YEAR == {"1d": 365.25, "1h": 8766.0}


def test_default_sharpe_annualization_is_sqrt_365() -> None:
    """THE pin: sharpe_ratio without a timeframe must scale by sqrt(365), not sqrt(8760).

    The expected value is computed with the same numpy operations on the same float64
    input as the implementation, so equality is exact -- no tolerance needed.
    """
    r = _DAILY.to_numpy(dtype="float64")
    expected = float(r.mean() / r.std(ddof=1) * np.sqrt(365))
    assert sharpe_ratio(_DAILY) == expected


def test_default_tstat_uses_365_25_calendar_days() -> None:
    """Pin the daily t-stat convention: t = SR_ann * sqrt(n_bars / 365.25).

    Both sides multiply the same two doubles (same sqrt input, IEEE-exact), so
    equality is exact -- no tolerance needed.
    """
    expected = sharpe_ratio(_DAILY) * float(np.sqrt(len(_DAILY) / 365.25))
    assert sharpe_tstat(_DAILY) == expected


def test_default_sortino_annualization_is_sqrt_365() -> None:
    """Pin sortino's default the same way: expected value computed with the same
    numpy operations on the same float64 input as the implementation, so equality
    is exact -- no tolerance needed."""
    r = _DAILY.to_numpy(dtype="float64")
    downside = np.minimum(r, 0.0)
    expected = float(r.mean() / np.sqrt(np.mean(downside**2)) * np.sqrt(365))
    assert sortino_ratio(_DAILY) == expected


def test_default_cagr_exponent_is_365_per_bar_count() -> None:
    # growth 4x over 730 bars: 4 ** (365/730) - 1 = 4 ** 0.5 - 1 = 1.0.
    # 365/730 == 0.5 and 4 ** 0.5 == 2.0 exactly in binary floating point,
    # so the assertion is exact -- no tolerance needed.
    returns = pd.Series([3.0] + [0.0] * 729)
    assert cagr(returns) == 1.0


def test_default_turnover_annualization_is_365() -> None:
    # 2 units traded over 4 bars -> 2 * 365 / 4 = 182.5, exact in binary
    # floating point (730/4 has an exact float64 representation).
    assert annualized_turnover(pd.Series([1.0, 0.0, 1.0, 0.0])) == 182.5


def test_hourly_tstat_equals_annualized_sharpe_times_sqrt_of_year_span() -> None:
    """1d-vs-1h coherence: on hourly bars, t == SR_ann * sqrt(span in calendar years).

    span_years = n_bars / (24 * 365.25): hourly bars per calendar year; 24 * 365.25
    == 8766.0 exactly in float64, so both sides divide by the same double and multiply
    the same two doubles. The difference is at most a few ulps (~1e-16 relative);
    1e-12 is that bound with four orders of magnitude of headroom.
    """
    rng = np.random.default_rng(7)
    r = pd.Series(rng.normal(0.0001, 0.01, size=5000))
    span_years = len(r) / (24.0 * 365.25)
    expected = sharpe_ratio(r, timeframe="1h") * math.sqrt(span_years)
    assert sharpe_tstat(r, timeframe="1h") == pytest.approx(expected, rel=1e-12)


def test_hourly_sharpe_is_daily_sharpe_times_sqrt_24() -> None:
    """Reading the same per-bar series as hourly scales the ratio by sqrt(8760/365).

    sqrt(8760) == sqrt(365 * 24) analytically; each floating sqrt is exactly rounded
    but sqrt is not multiplicative in floats, so the two sides differ by a few ulps
    (~1e-16 relative). 1e-12 is that bound with four orders of magnitude of headroom.
    """
    assert sharpe_ratio(_DAILY, timeframe="1h") == pytest.approx(
        sharpe_ratio(_DAILY) * math.sqrt(24.0), rel=1e-12
    )


def test_tstat_is_invariant_to_the_timeframe_declaration() -> None:
    """t = SR_bar * sqrt(n * 365/365.25) whatever the timeframe: the annualization
    factor cancels between SR_ann and n_years. This guards against double-counting
    the frequency -- the sqrt(24) class of bug the audit warned about.

    Analytic identity; the two sides differ only by floating rounding of
    intermediate sqrts (a few ulps, ~1e-16 relative), hence rel=1e-12.
    """
    assert sharpe_tstat(_DAILY, "1h") == pytest.approx(sharpe_tstat(_DAILY, "1d"), rel=1e-12)


def test_unknown_timeframe_is_rejected_with_explicit_error() -> None:
    with pytest.raises(ValueError, match="unsupported timeframe"):
        sharpe_ratio(_DAILY, timeframe="4h")
    with pytest.raises(ValueError, match="unsupported timeframe"):
        sharpe_tstat(_DAILY, timeframe="15m")
    with pytest.raises(ValueError, match="unsupported timeframe"):
        cagr(_DAILY, timeframe="1w")
    with pytest.raises(ValueError, match="unsupported timeframe"):
        sortino_ratio(_DAILY, timeframe="2h")
    with pytest.raises(ValueError, match="unsupported timeframe"):
        annualized_turnover(_DAILY, timeframe="5m")

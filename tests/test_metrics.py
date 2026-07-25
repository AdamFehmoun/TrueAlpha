"""One unit test per metric, against hand-computed values (not re-derived formulas)."""

from __future__ import annotations

import math

import pandas as pd
import pytest

from engine.metrics import (
    annualized_turnover,
    cagr,
    max_drawdown,
    sharpe_ratio,
    sharpe_tstat,
    sortino_ratio,
    total_return,
    win_rate,
)


def _s(values: list[float]) -> pd.Series:
    return pd.Series(values, dtype="float64")


def test_total_return() -> None:
    # 1.10 * 0.95 - 1 = 0.045
    assert total_return(_s([0.10, -0.05])) == pytest.approx(0.045, abs=1e-15)


def test_cagr_doubling_over_two_years_is_100pct_per_year() -> None:
    # growth 4x over 730 daily periods -> 4**(365/730) - 1 = 1.0
    returns = _s([3.0] + [0.0] * 729)
    assert cagr(returns, timeframe="1d") == pytest.approx(1.0, rel=1e-12)


def test_cagr_total_wipeout_is_minus_100pct() -> None:
    assert cagr(_s([0.5, -1.0])) == -1.0


def test_sharpe_hand_computed() -> None:
    # mean = 0.02, sample std (ddof=1) = 0.01 -> 2 * sqrt(365) = 38.2099...
    assert sharpe_ratio(_s([0.01, 0.02, 0.03]), timeframe="1d") == pytest.approx(38.2099, abs=1e-3)


def test_sharpe_zero_volatility_is_nan() -> None:
    assert math.isnan(sharpe_ratio(_s([0.01, 0.01, 0.01])))


def test_sharpe_tstat_hand_computed() -> None:
    # SR_ann = 38.2099 (see test_sharpe_hand_computed); n_years = 3/365.25
    # t = 38.2099 * sqrt(3/365.25) = 38.2099 * 0.0906286 = 3.4629
    assert sharpe_tstat(_s([0.01, 0.02, 0.03]), timeframe="1d") == pytest.approx(3.4629, abs=1e-3)


def test_sharpe_tstat_zero_volatility_is_nan() -> None:
    assert math.isnan(sharpe_tstat(_s([0.01, 0.01, 0.01])))


def test_sortino_hand_computed() -> None:
    # returns [0.02, -0.01, 0.03]: mean = 0.0133..., downside dev = sqrt(0.0001/3)
    # ratio = 2.3094, annualized = 2.3094 * 19.1050 = 44.122
    assert sortino_ratio(_s([0.02, -0.01, 0.03]), timeframe="1d") == pytest.approx(44.122, abs=1e-2)


def test_sortino_no_downside_positive_mean_is_inf() -> None:
    assert sortino_ratio(_s([0.01, 0.02])) == float("inf")


def test_sortino_no_downside_zero_mean_is_nan() -> None:
    assert math.isnan(sortino_ratio(_s([0.0, 0.0])))


def test_max_drawdown_hand_computed() -> None:
    # equity: 1.10, 0.55, 0.66 ; peak: 1.10 -> worst drawdown = 0.55/1.10 - 1 = -0.5
    assert max_drawdown(_s([0.10, -0.50, 0.20])) == pytest.approx(-0.5, abs=1e-15)


def test_max_drawdown_monotonic_up_is_zero() -> None:
    assert max_drawdown(_s([0.01, 0.02, 0.03])) == 0.0


def test_annualized_turnover_hand_computed() -> None:
    # total |trades| = 2 over 4 periods -> 2 * 365 / 4 = 182.5
    assert annualized_turnover(_s([1.0, 0.0, 1.0, 0.0]), timeframe="1d") == pytest.approx(
        182.5, abs=1e-12
    )


def test_win_rate_counts_only_active_periods() -> None:
    returns = _s([0.01, -0.02, 0.03, 0.0])
    positions = _s([1.0, 1.0, 0.0, 1.0])
    # active returns: [0.01, -0.02, 0.0] -> wins: 1/3
    assert win_rate(returns, positions) == pytest.approx(1.0 / 3.0, abs=1e-15)


def test_win_rate_never_active_is_nan() -> None:
    assert math.isnan(win_rate(_s([0.0, 0.0]), _s([0.0, 0.0])))

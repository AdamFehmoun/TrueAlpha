"""One unit test per metric, against hand-computed values (not re-derived formulas)."""

from __future__ import annotations

import math

import pandas as pd
import pytest

from engine.metrics import (
    annualized_turnover,
    cagr,
    flat_share,
    max_drawdown,
    sharpe_ratio,
    sharpe_standard_error,
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


def test_sharpe_standard_error_pins_the_oos_error_bar() -> None:
    """Lo (2002) SE of the annualized Sharpe on 648 daily bars = sqrt(365/648) = 0.7505.

    648 bars is exactly the repo's OOS window on the B4 span (round(3240 x 0.2)).
    The series alternates +/-1% over an EVEN count (324 up, 324 down), so the mean
    is exactly zero, Lo's finite-Sharpe correction vanishes, and the SE reduces to
    sqrt(365/648) = 0.75051... == 0.7505 within the 1e-3 pin. (The archived
    2022-2024 OOS window, 219 bars, corresponds to sqrt(365/219) = 1.291 -- the
    error bar the archived golden was published with.)"""
    returns = _s([0.01 if i % 2 == 0 else -0.01 for i in range(648)])
    assert sharpe_standard_error(returns, timeframe="1d") == pytest.approx(0.7505, abs=1e-3)


def test_sharpe_standard_error_zero_volatility_is_nan() -> None:
    assert math.isnan(sharpe_standard_error(_s([0.01, 0.01, 0.01])))


def test_tstat_equals_sharpe_over_se_with_explicit_corrections() -> None:
    """Internal coherence of t-stat and SE, EXACT at 1e-12 -- no fudge tolerance.

    Derivation (1d bars, n = len(returns), SR_bar = mean/std):
      Sharpe / SE = SR_bar*sqrt(365) / (sqrt(365) * sqrt((1 + SR_bar^2/2)/n))
                  = SR_bar * sqrt(n / (1 + SR_bar^2/2))
      t_stat      = SR_bar * sqrt(365) * sqrt(n/365.25) = SR_bar * sqrt(n * 365/365.25)
    Hence, with BOTH corrections written out explicitly,
      t_stat = (Sharpe / SE) * sqrt(365/365.25) * sqrt(1 + SR_bar^2/2)
    where sqrt(365/365.25) is the t-stat's calendar-year convention and
    sqrt(1 + SR_bar^2/2) is Lo's finite-Sharpe term. The identity is exact, so the
    tolerance is 1e-12; anything looser would be hiding an inconsistency."""
    returns = _s([0.013, -0.007, 0.021, 0.004, -0.016, 0.009, -0.002, 0.011])
    sr_bar = float(returns.mean() / returns.std(ddof=1))
    lhs = sharpe_tstat(returns, timeframe="1d")
    rhs = (
        sharpe_ratio(returns, timeframe="1d")
        / sharpe_standard_error(returns, timeframe="1d")
        * math.sqrt(365.0 / 365.25)
        * math.sqrt(1.0 + sr_bar**2 / 2.0)
    )
    assert lhs == pytest.approx(rhs, rel=1e-12)


def test_sharpe_standard_error_depends_only_on_calendar_span() -> None:
    """Same calendar span sampled 24x finer: the SE does NOT shrink.

    sqrt(A/n) is invariant when A and n scale together, so switching from daily to
    hourly bars buys no statistical significance -- the docstring's claim, enforced.
    Both series alternate +/-1% with an even length, so SR_bar is exactly 0 and both
    SEs reduce to sqrt(A/n) exactly; 100 days of daily bars vs the same 100 days of
    hourly bars must give the identical error bar."""
    daily = _s([0.01 if i % 2 == 0 else -0.01 for i in range(100)])
    hourly = _s([0.01 if i % 2 == 0 else -0.01 for i in range(100 * 24)])
    se_daily = sharpe_standard_error(daily, timeframe="1d")
    se_hourly = sharpe_standard_error(hourly, timeframe="1h")
    assert se_daily == pytest.approx(se_hourly, rel=1e-12)
    assert se_daily == pytest.approx(math.sqrt(365.0 / 100.0), rel=1e-12)


def test_flat_share_counts_zero_position_bars() -> None:
    assert flat_share(_s([0.0, 1.0, 0.0, 1.0])) == pytest.approx(0.5, abs=1e-15)
    assert flat_share(_s([1.0, 1.0])) == 0.0
    assert flat_share(_s([0.0, 0.0])) == 1.0


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

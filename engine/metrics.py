"""Performance metrics, each unit-tested against hand-computed values.

Conventions (documented once, applied everywhere):
- Metrics take the per-period NET returns series produced by the backtest.
- Frequency: every annualizing function takes an explicit ``timeframe`` parameter,
  defaulting to ``"1d"`` (daily bars). The timeframe string is the single source of
  truth for frequency, propagated from the data manifest's ``timeframe`` field; the
  supported values and their annualization factors live in ``BARS_PER_YEAR``
  (crypto trades around the clock: 365 daily bars or 8760 hourly bars per year).
- Ratios (Sharpe, Sortino) scale by sqrt(BARS_PER_YEAR[timeframe]).
- The t-stat converts the bar count into CALENDAR years via
  ``CALENDAR_BARS_PER_YEAR`` (= 365.25 x bars per day, leap-year-aware). This is the
  BAR-COUNT convention -- ``n_years = len(returns) / calendar_bars_per_year(tf)`` --
  not the real index span; the choice is documented in the README next to the numbers.
- Risk-free rate is 0.
- Sharpe uses the sample standard deviation (ddof=1); zero volatility -> NaN.
- Sortino's downside deviation is sqrt(mean(min(r, 0)^2)) over ALL periods (target 0);
  no downside -> +inf if mean return is positive, NaN otherwise.
- ``max_drawdown`` is reported as a negative number (e.g. -0.55 for -55%).
- ``win_rate`` counts only periods with a non-zero position (being flat is not a win).
"""

from __future__ import annotations

from typing import Final

import numpy as np
import pandas as pd

from engine.backtest import BacktestResult

DEFAULT_TIMEFRAME: Final = "1d"

_DAYS_PER_YEAR: Final = 365  # crypto trades every calendar day
_CALENDAR_DAYS_PER_YEAR: Final = 365.25  # mean calendar-year length (leap-year-aware)
_BARS_PER_DAY: Final[dict[str, int]] = {"1d": 1, "1h": 24}

# Annualization factor per timeframe: number of bars in a (trading) year.
BARS_PER_YEAR: Final[dict[str, int]] = {
    tf: _DAYS_PER_YEAR * bars for tf, bars in _BARS_PER_DAY.items()
}
# Bars per CALENDAR year, used ONLY to turn a bar count into a number of years for
# the t-stat. 365.25 preserves the pre-frequency-aware daily t-stat bit-for-bit.
CALENDAR_BARS_PER_YEAR: Final[dict[str, float]] = {
    tf: _CALENDAR_DAYS_PER_YEAR * bars for tf, bars in _BARS_PER_DAY.items()
}


def bars_per_year(timeframe: str) -> int:
    """Annualization factor for ``timeframe``: 365 for "1d", 8760 for "1h"."""
    try:
        return BARS_PER_YEAR[timeframe]
    except KeyError as exc:
        raise ValueError(
            f"unsupported timeframe {timeframe!r}; supported: {sorted(BARS_PER_YEAR)}"
        ) from exc


def calendar_bars_per_year(timeframe: str) -> float:
    """Bars per calendar year (365.25 x bars/day), used by the t-stat's n_years."""
    try:
        return CALENDAR_BARS_PER_YEAR[timeframe]
    except KeyError as exc:
        raise ValueError(
            f"unsupported timeframe {timeframe!r}; supported: {sorted(CALENDAR_BARS_PER_YEAR)}"
        ) from exc


def total_return(returns: pd.Series) -> float:
    r = returns.to_numpy(dtype="float64")
    return float(np.prod(1.0 + r) - 1.0)


def cagr(returns: pd.Series, timeframe: str = DEFAULT_TIMEFRAME) -> float:
    r = returns.to_numpy(dtype="float64")
    growth = float(np.prod(1.0 + r))
    if growth <= 0.0:
        return -1.0
    return float(growth ** (bars_per_year(timeframe) / len(r)) - 1.0)


def sharpe_ratio(returns: pd.Series, timeframe: str = DEFAULT_TIMEFRAME) -> float:
    r = returns.to_numpy(dtype="float64")
    periods = bars_per_year(timeframe)
    std = float(r.std(ddof=1))
    if std == 0.0:
        return float("nan")
    return float(r.mean() / std * np.sqrt(periods))


def sharpe_standard_error(returns: pd.Series, timeframe: str = DEFAULT_TIMEFRAME) -> float:
    """Lo (2002) standard error of the ANNUALIZED Sharpe ratio, i.i.d.-NORMAL returns.

    Per-bar: ``SE(SR_bar) = sqrt((1 + SR_bar^2 / 2) / n)``; annualized by
    ``sqrt(bars_per_year)``, the same factor as the Sharpe itself. Since the
    ``SR_bar^2 / 2`` term is negligible at bar scale, ``SE(SR_ann) ~= sqrt(A / n)
    = 1 / sqrt(years)``: the error bar on a Sharpe depends ONLY on the CALENDAR
    SPAN of the sample, not on the sampling frequency -- switching from daily to
    hourly bars multiplies A and n by the same 24 and shrinks nothing.

    Assumption scope, stated rather than hidden: the (1 + SR^2/2)/n variance is
    Lo's i.i.d.-normal result (the general i.i.d. case adds skew/kurtosis terms,
    Mertens 2002), and SERIAL DEPENDENCE -- e.g. the long/flat position
    persistence of the strategies in this repo -- WIDENS the true SE beyond this
    value. A wider true SE only strengthens a "not significant" conclusion, so
    this estimate is conservative for the no-edge findings it annotates; it is
    NOT conservative for declaring significance near the 1.96-SE boundary.
    Zero volatility -> NaN (no Sharpe, no error bar).
    """
    r = returns.to_numpy(dtype="float64")
    std = float(r.std(ddof=1))
    if std == 0.0:
        return float("nan")
    sr_bar = float(r.mean()) / std
    per_bar_se = np.sqrt((1.0 + sr_bar**2 / 2.0) / len(r))
    return float(per_bar_se * np.sqrt(bars_per_year(timeframe)))


def sharpe_tstat(returns: pd.Series, timeframe: str = DEFAULT_TIMEFRAME) -> float:
    """t-statistic of the annualized Sharpe: SR_ann * sqrt(n_years).

    ``n_years = len(returns) / calendar_bars_per_year(timeframe)`` -- the BAR-COUNT
    convention (not the index span), with a 365.25-day calendar year; for "1d" this
    is exactly the historical behavior (n_bars / 365.25), for "1h" it divides by
    8766 = 24 x 365.25. Rule of thumb: |t| < 2 means the Sharpe is not statistically
    distinguishable from 0.
    """
    sr = sharpe_ratio(returns, timeframe)
    if not np.isfinite(sr):
        return float("nan")
    n_years = len(returns) / calendar_bars_per_year(timeframe)
    return float(sr * np.sqrt(n_years))


def sortino_ratio(returns: pd.Series, timeframe: str = DEFAULT_TIMEFRAME) -> float:
    r = returns.to_numpy(dtype="float64")
    downside = np.minimum(r, 0.0)
    downside_dev = float(np.sqrt(np.mean(downside**2)))
    mean = float(r.mean())
    if downside_dev == 0.0:
        return float("inf") if mean > 0.0 else float("nan")
    return float(mean / downside_dev * np.sqrt(bars_per_year(timeframe)))


def max_drawdown(returns: pd.Series) -> float:
    r = returns.to_numpy(dtype="float64")
    equity = np.cumprod(1.0 + r)
    peak = np.maximum.accumulate(equity)
    return float(np.min(equity / peak - 1.0))


def annualized_turnover(turnover: pd.Series, timeframe: str = DEFAULT_TIMEFRAME) -> float:
    """Total traded notional (in units of equity) per year."""
    return float(turnover.sum() * bars_per_year(timeframe) / len(turnover))


def win_rate(returns: pd.Series, positions: pd.Series) -> float:
    active = positions.to_numpy(dtype="float64") != 0.0
    if not active.any():
        return float("nan")
    r = returns.to_numpy(dtype="float64")[active]
    return float((r > 0.0).mean())


def flat_share(positions: pd.Series) -> float:
    """Fraction of bars held with a ZERO position.

    Being flat is a real position (its zero returns belong in the series and depress
    measured volatility); this reports how much of the sample it covers, next to a
    win rate that -- by convention -- only counts the non-flat bars.
    """
    return float((positions.to_numpy(dtype="float64") == 0.0).mean())


def summarize(result: BacktestResult, timeframe: str = DEFAULT_TIMEFRAME) -> dict[str, float]:
    return {
        "total_return": total_return(result.returns),
        "cagr": cagr(result.returns, timeframe),
        "sharpe": sharpe_ratio(result.returns, timeframe),
        "sharpe_se": sharpe_standard_error(result.returns, timeframe),
        "t_stat": sharpe_tstat(result.returns, timeframe),
        "sortino": sortino_ratio(result.returns, timeframe),
        "max_drawdown": max_drawdown(result.returns),
        "annualized_turnover": annualized_turnover(result.turnover, timeframe),
        "win_rate": win_rate(result.returns, result.positions),
        "flat_share": flat_share(result.positions),
    }

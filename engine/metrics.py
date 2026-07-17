"""Performance metrics, each unit-tested against hand-computed values.

Conventions (documented once, applied everywhere):
- Metrics take the per-period NET returns series produced by the backtest.
- Annualization uses ``periods_per_year`` = 365 for daily crypto (markets trade every
  day), scaling ratios by sqrt(365).
- Risk-free rate is 0.
- Sharpe uses the sample standard deviation (ddof=1); zero volatility -> NaN.
- Sortino's downside deviation is sqrt(mean(min(r, 0)^2)) over ALL periods (target 0);
  no downside -> +inf if mean return is positive, NaN otherwise.
- ``max_drawdown`` is reported as a negative number (e.g. -0.55 for -55%).
- ``win_rate`` counts only periods with a non-zero position (being flat is not a win).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from engine.backtest import BacktestResult

PERIODS_PER_YEAR_CRYPTO = 365


def total_return(returns: pd.Series) -> float:
    r = returns.to_numpy(dtype="float64")
    return float(np.prod(1.0 + r) - 1.0)


def cagr(returns: pd.Series, periods_per_year: int = PERIODS_PER_YEAR_CRYPTO) -> float:
    r = returns.to_numpy(dtype="float64")
    growth = float(np.prod(1.0 + r))
    if growth <= 0.0:
        return -1.0
    return float(growth ** (periods_per_year / len(r)) - 1.0)


def sharpe_ratio(returns: pd.Series, periods_per_year: int = PERIODS_PER_YEAR_CRYPTO) -> float:
    r = returns.to_numpy(dtype="float64")
    std = float(r.std(ddof=1))
    if std == 0.0:
        return float("nan")
    return float(r.mean() / std * np.sqrt(periods_per_year))


def sortino_ratio(returns: pd.Series, periods_per_year: int = PERIODS_PER_YEAR_CRYPTO) -> float:
    r = returns.to_numpy(dtype="float64")
    downside = np.minimum(r, 0.0)
    downside_dev = float(np.sqrt(np.mean(downside**2)))
    mean = float(r.mean())
    if downside_dev == 0.0:
        return float("inf") if mean > 0.0 else float("nan")
    return float(mean / downside_dev * np.sqrt(periods_per_year))


def max_drawdown(returns: pd.Series) -> float:
    r = returns.to_numpy(dtype="float64")
    equity = np.cumprod(1.0 + r)
    peak = np.maximum.accumulate(equity)
    return float(np.min(equity / peak - 1.0))


def annualized_turnover(
    turnover: pd.Series, periods_per_year: int = PERIODS_PER_YEAR_CRYPTO
) -> float:
    """Total traded notional (in units of equity) per year."""
    return float(turnover.sum() * periods_per_year / len(turnover))


def win_rate(returns: pd.Series, positions: pd.Series) -> float:
    active = positions.to_numpy(dtype="float64") != 0.0
    if not active.any():
        return float("nan")
    r = returns.to_numpy(dtype="float64")[active]
    return float((r > 0.0).mean())


def summarize(
    result: BacktestResult, periods_per_year: int = PERIODS_PER_YEAR_CRYPTO
) -> dict[str, float]:
    return {
        "total_return": total_return(result.returns),
        "cagr": cagr(result.returns, periods_per_year),
        "sharpe": sharpe_ratio(result.returns, periods_per_year),
        "sortino": sortino_ratio(result.returns, periods_per_year),
        "max_drawdown": max_drawdown(result.returns),
        "annualized_turnover": annualized_turnover(result.turnover, periods_per_year),
        "win_rate": win_rate(result.returns, result.positions),
    }

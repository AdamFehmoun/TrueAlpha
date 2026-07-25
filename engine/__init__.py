"""Backtest engine: leak-safe splits, vectorized backtest, tested metrics."""

from engine.backtest import BacktestConfig, BacktestResult, run_backtest
from engine.metrics import (
    BARS_PER_YEAR,
    CALENDAR_BARS_PER_YEAR,
    DEFAULT_TIMEFRAME,
    annualized_turnover,
    bars_per_year,
    cagr,
    calendar_bars_per_year,
    max_drawdown,
    sharpe_ratio,
    sharpe_tstat,
    sortino_ratio,
    summarize,
    total_return,
    win_rate,
)
from engine.splits import (
    LeakageError,
    TemporalSplit,
    assert_no_leakage,
    temporal_train_test_split,
    walk_forward_splits,
)

__all__ = [
    "BARS_PER_YEAR",
    "CALENDAR_BARS_PER_YEAR",
    "DEFAULT_TIMEFRAME",
    "BacktestConfig",
    "BacktestResult",
    "LeakageError",
    "TemporalSplit",
    "annualized_turnover",
    "assert_no_leakage",
    "bars_per_year",
    "cagr",
    "calendar_bars_per_year",
    "max_drawdown",
    "run_backtest",
    "sharpe_ratio",
    "sharpe_tstat",
    "sortino_ratio",
    "summarize",
    "temporal_train_test_split",
    "total_return",
    "walk_forward_splits",
    "win_rate",
]

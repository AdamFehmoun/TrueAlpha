"""Backtest engine: leak-safe splits, vectorized backtest, tested metrics."""

from engine.backtest import BacktestConfig, BacktestResult, run_backtest
from engine.metrics import (
    CALENDAR_DAYS_PER_YEAR,
    PERIODS_PER_YEAR_CRYPTO,
    annualized_turnover,
    cagr,
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
)

__all__ = [
    "CALENDAR_DAYS_PER_YEAR",
    "PERIODS_PER_YEAR_CRYPTO",
    "BacktestConfig",
    "BacktestResult",
    "LeakageError",
    "TemporalSplit",
    "annualized_turnover",
    "assert_no_leakage",
    "cagr",
    "max_drawdown",
    "run_backtest",
    "sharpe_ratio",
    "sharpe_tstat",
    "sortino_ratio",
    "summarize",
    "temporal_train_test_split",
    "total_return",
    "win_rate",
]

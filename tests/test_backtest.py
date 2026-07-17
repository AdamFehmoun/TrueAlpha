"""Engine mechanics: execution timing (zero look-ahead), costs, marking, validation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from engine.backtest import BacktestConfig, run_backtest
from tests.utils import make_prices, random_walk_prices

ZERO_COST = BacktestConfig(fee_bps=0.0, slippage_bps=0.0)


def test_positions_are_lagged_signal() -> None:
    prices = random_walk_prices(50, seed=7)
    rng = np.random.default_rng(1)
    signal = pd.Series(
        rng.integers(0, 2, size=50).astype("float64"), index=prices.index, name="signal"
    )
    result = run_backtest(prices, signal, ZERO_COST)
    expected = signal.shift(1).fillna(0.0)
    pd.testing.assert_series_equal(result.positions, expected, check_names=False)


def test_no_lookahead_signal_at_close_t_fills_at_open_t_plus_1() -> None:
    # The +100% move open[2]->open[3] happens BEFORE the signal (fired at close[2])
    # can be executed. Only the open[3]->open[4] move (+100%) may be captured.
    prices = make_prices([100.0, 100.0, 100.0, 200.0, 400.0])
    signal = pd.Series([0.0, 0.0, 1.0, 1.0, 1.0], index=prices.index)
    result = run_backtest(prices, signal, ZERO_COST)
    assert result.equity.iloc[-1] == pytest.approx(2.0, rel=1e-12)  # 4.0 would be look-ahead


def test_round_trip_costs_are_exact() -> None:
    # Flat prices, one entry and one exit at 10 bps each: equity = 0.999^2 exactly.
    prices = make_prices([100.0] * 5)
    signal = pd.Series([0.0, 1.0, 0.0, 0.0, 0.0], index=prices.index)
    result = run_backtest(prices, signal, BacktestConfig(fee_bps=10.0, slippage_bps=0.0))
    assert result.equity.iloc[-1] == pytest.approx(0.999 * 0.999, abs=1e-15)


def test_slippage_adds_to_fees() -> None:
    prices = make_prices([100.0] * 4)
    signal = pd.Series([0.0, 1.0, 0.0, 0.0], index=prices.index)
    result = run_backtest(prices, signal, BacktestConfig(fee_bps=10.0, slippage_bps=5.0))
    assert result.equity.iloc[-1] == pytest.approx((1.0 - 0.0015) ** 2, abs=1e-15)


def test_final_bar_is_marked_at_its_close() -> None:
    prices = make_prices([100.0, 100.0], [100.0, 150.0])
    signal = pd.Series([1.0, 1.0], index=prices.index)
    result = run_backtest(prices, signal, ZERO_COST)
    assert result.equity.iloc[-1] == pytest.approx(1.5, rel=1e-12)


def test_flat_signal_means_flat_equity() -> None:
    prices = random_walk_prices(100, seed=3)
    signal = pd.Series(0.0, index=prices.index)
    result = run_backtest(prices, signal)
    assert (result.equity == 1.0).all()
    assert (result.turnover == 0.0).all()


def test_missing_column_raises() -> None:
    prices = make_prices([100.0, 101.0, 102.0]).drop(columns=["close"])
    signal = pd.Series(1.0, index=prices.index)
    with pytest.raises(ValueError, match="missing columns"):
        run_backtest(prices, signal)


def test_nan_price_raises() -> None:
    prices = make_prices([100.0, 101.0, 102.0])
    prices.loc[prices.index[1], "open"] = np.nan
    signal = pd.Series(1.0, index=prices.index)
    with pytest.raises(ValueError, match="NaN"):
        run_backtest(prices, signal)


def test_nan_signal_raises() -> None:
    prices = make_prices([100.0, 101.0, 102.0])
    signal = pd.Series([np.nan, 1.0, 1.0], index=prices.index)
    with pytest.raises(ValueError, match="NaN in signal"):
        run_backtest(prices, signal)


def test_mismatched_signal_index_raises() -> None:
    prices = make_prices([100.0, 101.0, 102.0])
    signal = pd.Series([1.0, 1.0], index=prices.index[:2])
    with pytest.raises(ValueError, match="index"):
        run_backtest(prices, signal)


def test_single_bar_raises() -> None:
    prices = make_prices([100.0])
    signal = pd.Series(1.0, index=prices.index)
    with pytest.raises(ValueError, match="at least 2 bars"):
        run_backtest(prices, signal)

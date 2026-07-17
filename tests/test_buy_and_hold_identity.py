"""THE critical engine guardrail.

Buy & hold through the engine MUST equal the asset's gross return net of the single
entry fee: equity_final = (1 - cost_rate) * close[-1] / open[1]
(first fill at open[1], held to the final close, no exit trade).

If this diverges, the engine has a bug (look-ahead, double-counted costs, wrong
compounding). Tested on synthetic random walks AND on the real committed data.
"""

from __future__ import annotations

import pandas as pd
import pytest

from data.loader import load_ohlcv
from engine.backtest import BacktestConfig, run_backtest
from strategies.buy_and_hold import buy_and_hold_signal
from tests.utils import random_walk_prices

STANDARD = BacktestConfig(fee_bps=10.0, slippage_bps=0.0)


def _expected_buy_and_hold_equity(prices: pd.DataFrame, cost_rate: float) -> float:
    return float((1.0 - cost_rate) * prices["close"].iloc[-1] / prices["open"].iloc[1])


@pytest.mark.parametrize("seed", [0, 1, 2, 42])
def test_identity_on_synthetic_random_walks(seed: int) -> None:
    prices = random_walk_prices(500, seed=seed)
    result = run_backtest(prices, buy_and_hold_signal(prices), STANDARD)
    expected = _expected_buy_and_hold_equity(prices, STANDARD.cost_rate)
    assert result.equity.iloc[-1] == pytest.approx(expected, rel=1e-12)


@pytest.mark.parametrize("symbol", ["BTC/USDT", "ETH/USDT"])
def test_identity_on_real_data(symbol: str) -> None:
    prices = load_ohlcv(symbol)  # hash-verified load
    result = run_backtest(prices, buy_and_hold_signal(prices), STANDARD)
    expected = _expected_buy_and_hold_equity(prices, STANDARD.cost_rate)
    assert result.equity.iloc[-1] == pytest.approx(expected, rel=1e-12)


def test_identity_with_zero_costs_is_pure_gross_return() -> None:
    prices = random_walk_prices(300, seed=9)
    result = run_backtest(
        prices, buy_and_hold_signal(prices), BacktestConfig(fee_bps=0.0, slippage_bps=0.0)
    )
    expected = _expected_buy_and_hold_equity(prices, 0.0)
    assert result.equity.iloc[-1] == pytest.approx(expected, rel=1e-12)

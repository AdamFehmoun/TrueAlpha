"""Strategy contracts: causality (no future data), warmup handling, basic shape."""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd
import pytest

from strategies import buy_and_hold_signal, ma_crossover_signal
from tests.utils import random_walk_prices

Strategy = Callable[[pd.DataFrame], pd.Series]

STRATEGIES: list[tuple[str, Strategy]] = [
    ("buy_and_hold", buy_and_hold_signal),
    ("ma_crossover", ma_crossover_signal),
]


@pytest.mark.parametrize(("name", "strategy"), STRATEGIES)
def test_signal_is_causal(name: str, strategy: Strategy) -> None:
    """signal[t] must not change when future bars are appended."""
    prices = random_walk_prices(200, seed=11)
    full = strategy(prices)
    prefix = strategy(prices.iloc[:150])
    pd.testing.assert_series_equal(full.iloc[:150], prefix, check_names=False)


@pytest.mark.parametrize(("name", "strategy"), STRATEGIES)
def test_signal_matches_price_index_and_has_no_nan(name: str, strategy: Strategy) -> None:
    prices = random_walk_prices(120, seed=5)
    signal = strategy(prices)
    assert signal.index.equals(prices.index)
    assert not signal.isna().any()


def test_buy_and_hold_is_always_fully_long() -> None:
    prices = random_walk_prices(60, seed=2)
    assert (buy_and_hold_signal(prices) == 1.0).all()


def test_ma_crossover_is_flat_during_warmup() -> None:
    prices = random_walk_prices(120, seed=8)
    signal = ma_crossover_signal(prices, fast=20, slow=50)
    assert (signal.iloc[:49] == 0.0).all()


def test_ma_crossover_is_long_flat_only() -> None:
    prices = random_walk_prices(300, seed=13)
    signal = ma_crossover_signal(prices)
    assert set(signal.unique()) <= {0.0, 1.0}


def test_ma_crossover_rejects_fast_gte_slow() -> None:
    prices = random_walk_prices(60, seed=1)
    with pytest.raises(ValueError, match="fast"):
        ma_crossover_signal(prices, fast=50, slow=20)

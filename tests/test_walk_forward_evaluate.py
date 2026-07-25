"""The two invariants that make "OOS" honest, plus the evaluation pipeline contract.

1. Selection is BLIND to the test window: perturbing prices after a fold's train
   changes no selected parameter on that fold (or any earlier one).
2. Measurement ignores future bars: truncating the series after a fold's test window
   changes nothing in that fold, bit-for-bit.

Plus: the leakage validator runs INSIDE the pipeline; the argmax runs on train (a
rigged fixture where train and test disagree proves it); NaN candidates never win;
look-back bars feed the strategy but never the selection; explicit errors everywhere.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd
import pytest

from engine.backtest import BacktestConfig
from engine.evaluate import walk_forward_evaluate
from engine.metrics import sharpe_ratio
from engine.splits import LeakageError, TemporalSplit, walk_forward_splits
from strategies import ma_crossover_signal
from tests.utils import make_prices, random_walk_prices

COSTS = BacktestConfig(fee_bps=10.0, slippage_bps=0.0)
ZERO_COST = BacktestConfig(fee_bps=0.0, slippage_bps=0.0)

GRID: list[dict[str, int]] = [
    {"fast": 5, "slow": 20},
    {"fast": 10, "slow": 30},
    {"fast": 20, "slow": 60},
]


def _ma_factory(prices: pd.DataFrame, params: Mapping[str, Any]) -> pd.Series:
    return ma_crossover_signal(prices, fast=int(params["fast"]), slow=int(params["slow"]))


def _directional_factory(prices: pd.DataFrame, params: Mapping[str, Any]) -> pd.Series:
    # constant long (+1) or short (-1): the train argmax is fully predictable
    return pd.Series(float(params["side"]), index=prices.index, name="signal")


def _flat_or_long_factory(prices: pd.DataFrame, params: Mapping[str, Any]) -> pd.Series:
    value = 0.0 if params["mode"] == "flat" else 1.0
    return pd.Series(value, index=prices.index, name="signal")


# --------------------------------------------------------------------------- #
# the two invariants
# --------------------------------------------------------------------------- #


def test_selection_is_blind_to_test_window() -> None:
    """Perturbing every bar after a fold's train (gap + test + beyond) by x1.5 must
    change no selected parameter on that fold or any earlier one. If it does, the
    selection sees the future. (Later folds legitimately train on earlier folds'
    test bars -- that is what walk-forward means -- so only folds <= k are pinned.)
    """
    prices = random_walk_prices(400, seed=11)
    splits = walk_forward_splits(400, n_folds=3, test_size=0.3, purge=60, embargo=2)
    base = walk_forward_evaluate(prices, _ma_factory, GRID, splits, COSTS)
    for k, split in enumerate(splits):
        tampered = prices.copy()
        cutoff = int(split.train[-1]) + 1
        tampered.iloc[cutoff:] = tampered.iloc[cutoff:] * 1.5
        result = walk_forward_evaluate(tampered, _ma_factory, GRID, splits, COSTS)
        for j in range(k + 1):
            assert result.folds[j].params == base.folds[j].params
            assert result.folds[j].train_metric == base.folds[j].train_metric


def test_fold_test_returns_ignore_future_bars() -> None:
    """Truncating the series right after a fold's test window changes NOTHING in
    that fold or earlier ones: same parameters, same train metric, and bit-for-bit
    identical test returns. Each window's last bar is marked at its own close by
    construction, so no next-open dependency exists to excuse a difference."""
    prices = random_walk_prices(400, seed=5)
    splits = walk_forward_splits(400, n_folds=3, test_size=0.3, purge=60, embargo=2)
    full = walk_forward_evaluate(prices, _ma_factory, GRID, splits, COSTS)
    for k, split in enumerate(splits):
        truncated = prices.iloc[: int(split.test[-1]) + 1]
        result = walk_forward_evaluate(truncated, _ma_factory, GRID, list(splits[: k + 1]), COSTS)
        for j in range(k + 1):
            assert result.folds[j].params == full.folds[j].params
            assert result.folds[j].train_metric == full.folds[j].train_metric
            pd.testing.assert_series_equal(  # exact: no tolerance
                result.folds[j].test_result.returns,
                full.folds[j].test_result.returns,
                check_exact=True,
            )


# --------------------------------------------------------------------------- #
# the validator runs inside the pipeline
# --------------------------------------------------------------------------- #


def test_pipeline_rejects_a_tampered_split() -> None:
    prices = random_walk_prices(400, seed=3)
    good = walk_forward_splits(400, n_folds=3, test_size=0.3, purge=60, embargo=2)
    bad = TemporalSplit(  # train stretched to 1 bar before test: gap 1 < purge+embargo
        train=np.arange(0, int(good[1].test.min()) - 1, dtype=np.int64),
        test=good[1].test,
        purge=good[1].purge,
        embargo=good[1].embargo,
    )
    with pytest.raises(LeakageError, match="gap"):
        walk_forward_evaluate(prices, _ma_factory, GRID, [good[0], bad], COSTS)


def test_out_of_order_test_windows_across_folds_are_rejected() -> None:
    prices = random_walk_prices(400, seed=1)
    splits = walk_forward_splits(400, n_folds=3, test_size=0.3, purge=30, embargo=0)
    with pytest.raises(ValueError, match="chronological"):
        walk_forward_evaluate(prices, _ma_factory, GRID, [splits[1], splits[0], splits[2]], COSTS)


# --------------------------------------------------------------------------- #
# the argmax runs on train, with deterministic NaN/tie handling
# --------------------------------------------------------------------------- #


def test_selection_maximizes_the_train_metric_not_the_test() -> None:
    """Prices fall over the train window and rise over the test window: the train
    argmax picks the SHORT side even though LONG wins on test -- and the OOS return
    is then honestly negative. If the argmax ever peeked at test, this would flip."""
    rng = np.random.default_rng(2)
    steps = np.concatenate([rng.normal(-0.004, 0.002, 260), rng.normal(0.004, 0.002, 140)])
    prices = make_prices((100.0 * np.cumprod(1.0 + steps)).tolist())
    splits = walk_forward_splits(400, n_folds=1, test_size=0.25, purge=10, embargo=0)
    result = walk_forward_evaluate(
        prices,
        _directional_factory,
        [{"side": 1.0}, {"side": -1.0}],  # LONG listed first: SHORT must win on merit
        splits,
        ZERO_COST,
    )
    assert result.folds[0].params == {"side": -1.0}
    assert result.metrics["total_return"] < 0.0


def test_nan_selection_scores_never_win() -> None:
    # flat -> zero-volatility -> NaN sharpe; listed FIRST but must lose to long
    prices = random_walk_prices(200, seed=9)
    splits = walk_forward_splits(200, n_folds=1, test_size=0.2)
    result = walk_forward_evaluate(
        prices,
        _flat_or_long_factory,
        [{"mode": "flat"}, {"mode": "long"}],
        splits,
        ZERO_COST,
    )
    assert result.folds[0].params == {"mode": "long"}


def test_all_nan_selection_raises_explicitly() -> None:
    prices = random_walk_prices(200, seed=9)
    splits = walk_forward_splits(200, n_folds=1, test_size=0.2)
    with pytest.raises(ValueError, match="finite"):
        walk_forward_evaluate(prices, _flat_or_long_factory, [{"mode": "flat"}], splits)


# --------------------------------------------------------------------------- #
# look-back is fed to the strategy; the OOS series is an honest concatenation
# --------------------------------------------------------------------------- #


def test_test_window_signal_uses_lookback_bars_before_the_window() -> None:
    """A slow MA longer than the test window would leave a COLD window flat forever;
    the evaluator must feed pre-window history to the strategy (that is look-back,
    not leakage), so in-window positions can be long from the second bar on."""
    prices = make_prices((100.0 * np.cumprod(1.0 + np.full(400, 0.003))).tolist())
    splits = walk_forward_splits(400, n_folds=1, test_size=0.1, purge=60, embargo=0)
    result = walk_forward_evaluate(
        prices, _ma_factory, [{"fast": 5, "slow": 60}], splits, ZERO_COST
    )
    positions = result.folds[0].test_result.positions  # 40-bar window, slow=60
    assert (positions.iloc[1:] == 1.0).all()


def test_oos_series_is_the_chronological_concatenation_of_fold_tests() -> None:
    prices = random_walk_prices(400, seed=13)
    splits = walk_forward_splits(400, n_folds=4, test_size=0.3, purge=30, embargo=2)
    result = walk_forward_evaluate(prices, _ma_factory, GRID, splits, COSTS)

    expected = pd.concat([f.test_result.returns for f in result.folds])
    pd.testing.assert_series_equal(result.oos.returns, expected, check_exact=True)
    assert result.oos.returns.index.is_monotonic_increasing
    assert len(result.oos.returns) == sum(len(s.test) for s in splits)

    # metrics come from the concatenated series itself: identical computation,
    # identical float (the fixture is chosen so the OOS sharpe is finite)
    oos_sharpe = sharpe_ratio(result.oos.returns)
    assert math.isfinite(oos_sharpe)
    assert result.metrics["sharpe"] == oos_sharpe

    # OOS equity compounds across folds, each fold starting flat at 1.0
    pd.testing.assert_series_equal(
        result.oos.equity, (1.0 + expected).cumprod(), check_exact=True, check_names=False
    )


# --------------------------------------------------------------------------- #
# explicit errors
# --------------------------------------------------------------------------- #


def test_empty_grid_or_splits_raise() -> None:
    prices = random_walk_prices(200, seed=0)
    splits = walk_forward_splits(200, n_folds=1, test_size=0.2)
    with pytest.raises(ValueError, match="param_grid"):
        walk_forward_evaluate(prices, _ma_factory, [], splits)
    with pytest.raises(ValueError, match="splits"):
        walk_forward_evaluate(prices, _ma_factory, GRID, [])


def test_unknown_selection_metric_raises() -> None:
    prices = random_walk_prices(200, seed=0)
    splits = walk_forward_splits(200, n_folds=1, test_size=0.2)
    with pytest.raises(ValueError, match="selection_metric"):
        walk_forward_evaluate(prices, _ma_factory, GRID, splits, selection_metric="alpha")

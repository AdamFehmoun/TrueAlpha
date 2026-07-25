"""Walk-forward out-of-sample evaluation: selection on train, measurement on test.

The rule that makes the word "OOS" honest
-----------------------------------------
- The SIGNAL is causal: signal[t] depends only on bars <= t (proven by
  tests/test_strategies.py::test_signal_is_causal). Bars BEFORE a window feed the
  strategy's look-back (an MA(50) needs 50 bars of history); that is a look-back
  window, not leakage.
- PARAMETER SELECTION is confined to the train slice: every candidate is backtested
  on the train window only, from a price history TRUNCATED at the end of train. The
  selection step is never shown a single bar past the end of train, so even a
  non-causal strategy could not leak the test window into the choice of parameters.
- MEASUREMENT is confined to the test slice: the winning candidate is backtested on
  the test window only, from a history truncated at the end of test.

Window backtests are standalone: each window starts flat (position 0), enters at the
first bar its in-window signal dictates, and its last bar is marked at that bar's own
close (``run_backtest``'s convention) -- never at an open beyond the window. The OOS
equity is therefore the product of the fold equities, each fold starting at 1.0.

``assert_no_leakage`` runs INSIDE this pipeline for every fold -- not only in tests.
DoD criterion 4 demands that a strategy evaluation PASS the validator, not merely
that the validator exist.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

import numpy as np
import pandas as pd

from engine.backtest import BacktestConfig, BacktestResult, run_backtest
from engine.metrics import DEFAULT_TIMEFRAME, sharpe_ratio, sortino_ratio, summarize
from engine.metrics import total_return as _total_return
from engine.splits import TemporalSplit, assert_no_leakage

Params = Mapping[str, Any]
StrategyFactory = Callable[[pd.DataFrame, Params], pd.Series]

SELECTION_METRICS: Final[dict[str, Callable[[pd.Series, str], float]]] = {
    "sharpe": sharpe_ratio,
    "sortino": sortino_ratio,
    "total_return": lambda returns, _timeframe: _total_return(returns),
}


@dataclass(frozen=True)
class FoldResult:
    fold: int
    split: TemporalSplit
    params: dict[str, Any]
    train_metric: float  # selection metric of the winner on ITS train slice
    test_result: BacktestResult  # standalone backtest of the winner on the test slice


@dataclass(frozen=True)
class WalkForwardResult:
    folds: list[FoldResult]
    oos: BacktestResult  # chronological concatenation of the fold test slices
    metrics: dict[str, float]  # summarize(oos, timeframe)
    selection_metric: str
    timeframe: str


def _window_backtest(
    prices: pd.DataFrame,
    factory: StrategyFactory,
    params: Params,
    window: np.ndarray[Any, np.dtype[np.int64]],
    config: BacktestConfig,
) -> BacktestResult:
    """Backtest one candidate on one window, history truncated at the window's end.

    The strategy sees ``prices.iloc[:window[-1] + 1]`` -- look-back allowed, future
    denied by construction -- then both prices and signal are sliced to the window.
    """
    history = prices.iloc[: int(window[-1]) + 1]
    signal = factory(history, params)
    if not signal.index.equals(history.index):
        raise ValueError("strategy signal index must match its price history index")
    return run_backtest(prices.iloc[window], signal.iloc[window], config)


def walk_forward_evaluate(
    prices: pd.DataFrame,
    strategy_factory: StrategyFactory,
    param_grid: Sequence[Params],
    splits: Sequence[TemporalSplit],
    config: BacktestConfig | None = None,
    selection_metric: str = "sharpe",
    timeframe: str = DEFAULT_TIMEFRAME,
) -> WalkForwardResult:
    """Evaluate a strategy family strictly out-of-sample over walk-forward folds.

    Per fold, in this order:
    1. ``assert_no_leakage(train, test, purge + embargo)`` -- in the pipeline itself;
    2. every grid candidate is backtested on the TRAIN slice only -> selection metric;
    3. argmax on train (NaN scores can never win; ties break by grid order) -> ONE
       parameter set for this fold;
    4. that parameter set is backtested on the TEST slice only -> the fold's OOS
       returns.

    The K fold test slices are then concatenated chronologically into the OOS series
    on which ``metrics`` are computed. Raises an explicit error instead of guessing:
    empty grid/splits, unknown metric, out-of-order or overlapping test windows, or
    a fold where no candidate has a finite selection metric.
    """
    if len(param_grid) == 0:
        raise ValueError("param_grid is empty")
    if len(splits) == 0:
        raise ValueError("splits is empty")
    try:
        metric_fn = SELECTION_METRICS[selection_metric]
    except KeyError as exc:
        raise ValueError(
            f"unknown selection_metric {selection_metric!r}; supported: {sorted(SELECTION_METRICS)}"
        ) from exc
    cfg = config if config is not None else BacktestConfig()

    # test windows must tile forward across folds, or the OOS concatenation would
    # double-count or reorder bars
    all_test = np.concatenate([np.asarray(s.test) for s in splits])
    if not bool((np.diff(all_test) > 0).all()):
        raise ValueError("test windows must be chronological and disjoint across folds")

    folds: list[FoldResult] = []
    for k, split in enumerate(splits):
        # the validator runs INSIDE the pipeline for every fold (DoD criterion 4)
        assert_no_leakage(split.train, split.test, min_gap=split.purge + split.embargo)

        selected: dict[str, Any] | None = None
        selected_score = float("nan")
        for params in param_grid:
            train_result = _window_backtest(prices, strategy_factory, params, split.train, cfg)
            score = metric_fn(train_result.returns, timeframe)
            # NaN (e.g. a candidate flat over the whole train) can never win the
            # argmax; ties break by grid order (first candidate wins) for determinism
            if math.isnan(score):
                continue
            if selected is None or score > selected_score:
                selected = dict(params)
                selected_score = score
        if selected is None:
            raise ValueError(
                f"fold {k}: no grid candidate produced a finite {selection_metric!r} "
                "on train; refusing to select a parameter set arbitrarily"
            )

        test_result = _window_backtest(prices, strategy_factory, selected, split.test, cfg)
        folds.append(
            FoldResult(
                fold=k,
                split=split,
                params=selected,
                train_metric=selected_score,
                test_result=test_result,
            )
        )

    returns = pd.concat([f.test_result.returns for f in folds])
    positions = pd.concat([f.test_result.positions for f in folds])
    turnover = pd.concat([f.test_result.turnover for f in folds])
    equity = (1.0 + returns).cumprod()
    oos = BacktestResult(
        equity=equity.rename("equity"),
        returns=returns.rename("returns"),
        positions=positions.rename("positions"),
        turnover=turnover.rename("turnover"),
    )
    return WalkForwardResult(
        folds=folds,
        oos=oos,
        metrics=summarize(oos, timeframe),
        selection_metric=selection_metric,
        timeframe=timeframe,
    )

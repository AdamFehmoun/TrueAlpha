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
- MEASUREMENT is confined to the OOS region: the winning candidate is backtested on
  test bars only, from a history truncated at the end of its SEGMENT (the maximal
  run of same-parameter contiguous folds it belongs to -- see the boundary rule
  below), never beyond the OOS region. For a causal strategy, per-fold measurement
  differs from a per-fold truncation only in the boundary marking, proven
  bar-for-bar by test_fold_test_returns_ignore_future_bars.

Fold splicing (the boundary rule, decided by test_adjacent_folds_splice_exactly)
--------------------------------------------------------------------------------
Consecutive folds that select IDENTICAL parameters and whose test windows are
contiguous form a SEGMENT, backtested as ONE window: the end-of-fold position
carries into the next fold, so the boundary adds no artificial flat bar and no
phantom re-entry fee. A real parameter change (or a gap between test windows)
starts a new segment, which begins flat -- re-allocating to a new parameterization
is actual behavior, not a splice artifact. Each segment starts flat (position 0),
enters at the first bar its in-window signal dictates, and its LAST bar is marked
at that bar's own close (``run_backtest``'s convention) -- never at an open beyond
the segment. Per-fold results are slices of their segment's run (an interior
fold's last bar is therefore marked at the next fold's first open -- the honest
mark of a continuously held position); each fold's reported equity restarts at 1.
A segment's outgoing position at a parameter change is NOT charged an exit fee:
the engine never bills terminal liquidation (the same convention that makes the
buy-and-hold identity "gross return net of the single entry fee"), so across such
a boundary the positions series can step to 0 with no turnover entry. This case
DOES occur on the published runs since ``9a7fd1c`` (B4-B): BTC/USDT rolling
selects 5/50 on fold 1 and 5/100 on folds 2-5, so the boundary carries an
outgoing position of 1.0 that is never charged an exit fee.
``exit_fee_bias_bps`` quantifies it exactly (12.3939 bps there); B10 tracks the
timing convention.

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
    # count of fold boundaries that actually CARRIED a non-zero position into the
    # next fold (identical selected parameters + contiguous windows + non-zero
    # position at the boundary). This number is what moved the OOS figures when
    # splicing was introduced, so it is pinned here, published, and tested --
    # never inferred by hand.
    n_carried_boundaries: int
    # temporal coverage of the OOS series, engine-counted (single source, like
    # n_carried_boundaries): the published Sharpe includes flat bars as zero
    # returns, so the reader needs these to recompute the exposed-only variant
    n_bars_exposed: int  # OOS bars held with a non-zero position
    time_in_market: float  # n_bars_exposed / len(oos.returns)
    # segment accounting for the unbilled-exit bias, engine-counted:
    n_segments: int
    n_uncharged_exits: int  # segments whose outgoing position is non-zero
    # EXACT effect on total_return of the unbilled exits, computed by re-billing
    # the missing liquidation at the last bar of each concerned segment (exact in
    # amount; one bar early in timing versus a continuous book)
    exit_fee_bias_bps: float


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


# pandas frequency of one bar per timeframe, for the calendar-aware hole check
_BAR_FREQ: Final[dict[str, str]] = {"1d": "D", "1h": "h"}


def _assert_window_calendar(
    prices: pd.DataFrame,
    window: np.ndarray[Any, np.dtype[np.int64]],
    timeframe: str,
    known_holes: pd.DatetimeIndex | None,
    fold: int,
    window_name: str,
) -> None:
    """Reject any calendar gap inside a window that is not a whitelisted hole.

    ``known_holes`` is the manifest's pinned ``missing_bars`` list (empty or None
    for gap-free data): a hole in the whitelist is accepted -- the position is held
    across it and the pre-hole bar is marked at the next available open, which is
    real market movement, not an invented return. Any other calendar gap fails
    loudly and is listed by timestamp, exactly like the data loader does.
    """
    try:
        freq = _BAR_FREQ[timeframe]
    except KeyError as exc:
        raise ValueError(
            f"unsupported timeframe {timeframe!r}; supported: {sorted(_BAR_FREQ)}"
        ) from exc
    idx = pd.DatetimeIndex(prices.index[window])
    expected = pd.date_range(start=idx[0], end=idx[-1], freq=freq)
    missing = expected.difference(idx)
    if len(missing) == 0:
        return
    whitelist = known_holes if known_holes is not None else pd.DatetimeIndex([])
    unknown = missing.difference(whitelist)
    if len(unknown) > 0:
        listed = ", ".join(ts.isoformat() for ts in unknown[:10])
        raise ValueError(
            f"fold {fold}: {window_name} window skips {len(unknown)} calendar bar(s) "
            f"not whitelisted as known holes: {listed}" + ("…" if len(unknown) > 10 else "")
        )


def walk_forward_evaluate(
    prices: pd.DataFrame,
    strategy_factory: StrategyFactory,
    param_grid: Sequence[Params],
    splits: Sequence[TemporalSplit],
    config: BacktestConfig | None = None,
    selection_metric: str = "sharpe",
    timeframe: str = DEFAULT_TIMEFRAME,
    known_holes: pd.DatetimeIndex | None = None,
) -> WalkForwardResult:
    """Evaluate a strategy family strictly out-of-sample over walk-forward folds.

    Per fold, in this order:
    1. ``assert_no_leakage(train, test, purge + embargo)`` -- in the pipeline itself;
    2. every grid candidate is backtested on the TRAIN slice only -> selection metric;
    3. argmax on train (NaN scores can never win; ties break by grid order) -> ONE
       parameter set for this fold;
    4. that parameter set is backtested on the TEST region, spliced by segments:
       consecutive folds with identical parameters and contiguous test windows run
       as ONE backtest (position carries over the boundary), and a parameter change
       resets to flat (see the module docstring's boundary rule).

    Hole policy (``known_holes``): the validator is CALENDAR-aware. A window whose
    calendar gaps are ALL in ``known_holes`` -- the manifest's pinned
    ``missing_bars`` whitelist, generated by the ingestion script and cross-checked
    against the hashed data by the loader -- is accepted: the position is held
    across the hole and the next available open marks the pre-hole bar (real market
    movement; no bar interpolated, no return invented). Any calendar gap NOT in the
    whitelist fails loudly, listed by timestamp. Positionally gapped windows
    (excluding bars that EXIST in the frame) are rejected unconditionally.

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

    picks: list[tuple[dict[str, Any], float]] = []
    for k, split in enumerate(splits):
        # the validator runs INSIDE the pipeline for every fold (DoD criterion 4)
        assert_no_leakage(split.train, split.test, min_gap=split.purge + split.embargo)

        # the vectorized engine marks bar t at open[t+1] WITHIN its window: a gapped
        # window would silently compress the whole gap move into the pre-gap bar and
        # bill it to the strategy as if the position were held across excluded bars.
        # Non-contiguous windows are rejected loudly instead of mispriced silently.
        for window_name, window in (("train", split.train), ("test", split.test)):
            if not bool((np.diff(np.asarray(window)) == 1).all()):
                raise ValueError(
                    f"fold {k}: {window_name} window is not contiguous; the engine's "
                    "next-open marking cannot price across excluded bars"
                )
            _assert_window_calendar(prices, window, timeframe, known_holes, k, window_name)

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
        picks.append((selected, selected_score))

    # splice rule: consecutive folds with identical parameters and contiguous test
    # windows run as ONE backtest (the position carries over the fold boundary); a
    # parameter change or a window gap starts a new segment, which begins flat
    segments: list[list[int]] = [[0]]
    for k in range(1, len(splits)):
        same_params = picks[k][0] == picks[k - 1][0]
        contiguous = int(splits[k].test[0]) == int(splits[k - 1].test[-1]) + 1
        if same_params and contiguous:
            segments[-1].append(k)
        else:
            segments.append([k])

    folds: list[FoldResult] = []
    n_carried = 0
    for segment in segments:
        window = np.concatenate([np.asarray(splits[k].test) for k in segment])
        # the splice rule only ever joins contiguous windows: VERIFY it instead of
        # assuming it (belt-and-braces; must stay green)
        if not bool((np.diff(window) == 1).all()):
            raise ValueError(
                f"segment {segment}: concatenated window is not contiguous; "
                "the splice rule must never join non-adjacent test windows"
            )
        # re-validate the calendar of the UNION: a hole sitting exactly at a fold
        # boundary is internal to neither per-fold window (both per-fold checks
        # pass) but IS internal to the spliced window -- same structural class as
        # mutation M7: per-fold checks, unverified concatenation
        _assert_window_calendar(prices, window, timeframe, known_holes, segment[0], "segment")
        seg_result = _window_backtest(prices, strategy_factory, picks[segment[0]][0], window, cfg)
        offset = 0
        for k in segment:
            if k != segment[0] and float(seg_result.positions.iloc[offset]) != 0.0:
                # a within-segment boundary whose position was actually non-zero:
                # this boundary CARRIED, i.e. the splice rule changed its accounting
                n_carried += 1
            n_k = len(splits[k].test)
            fold_returns = seg_result.returns.iloc[offset : offset + n_k]
            folds.append(
                FoldResult(
                    fold=k,
                    split=splits[k],
                    params=dict(picks[k][0]),
                    train_metric=picks[k][1],
                    test_result=BacktestResult(
                        # fold-local equity restarts at 1; returns/positions/turnover
                        # are the fold's exact slice of its segment's run
                        equity=(1.0 + fold_returns).cumprod().rename("equity"),
                        returns=fold_returns,
                        positions=seg_result.positions.iloc[offset : offset + n_k],
                        turnover=seg_result.turnover.iloc[offset : offset + n_k],
                    ),
                )
            )
            offset += n_k

    returns = pd.concat([f.test_result.returns for f in folds])
    positions = pd.concat([f.test_result.positions for f in folds])
    turnover = pd.concat([f.test_result.turnover for f in folds])
    equity = (1.0 + returns).cumprod()

    # temporal coverage (single source: consumers must not re-count)
    n_bars_exposed = int((positions.to_numpy(dtype="float64") != 0.0).sum())

    # EXACT unbilled-exit bias: re-bill the missing liquidation at the last bar of
    # each segment whose outgoing position is non-zero, recompound, and take the
    # difference of the totals. Exact in amount; one bar early in timing versus a
    # continuous book, which would bill it at the next segment's first bar.
    extra = pd.Series(0.0, index=positions.index)
    n_uncharged = 0
    offset = 0
    for segment in segments:
        n_seg = sum(len(splits[k].test) for k in segment)
        outgoing = float(positions.iloc[offset + n_seg - 1])
        if outgoing != 0.0:
            n_uncharged += 1
            extra.iloc[offset + n_seg - 1] += abs(outgoing)
        offset += n_seg
    net_liq = (1.0 + returns) * (1.0 - extra * cfg.cost_rate) - 1.0
    exit_fee_bias_bps = (float((1.0 + returns).prod()) - float((1.0 + net_liq).prod())) * 1e4
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
        n_carried_boundaries=n_carried,
        n_bars_exposed=n_bars_exposed,
        time_in_market=n_bars_exposed / len(returns),
        n_segments=len(segments),
        n_uncharged_exits=n_uncharged,
        exit_fee_bias_bps=exit_fee_bias_bps,
    )

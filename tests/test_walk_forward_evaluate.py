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

from engine.backtest import BacktestConfig, run_backtest
from engine.evaluate import walk_forward_evaluate
from engine.metrics import sharpe_ratio
from engine.splits import LeakageError, TemporalSplit, walk_forward_splits
from strategies import ma_crossover_signal
from tests.utils import make_prices, random_walk_prices, regime_shift_prices

COSTS = BacktestConfig(fee_bps=10.0, slippage_bps=0.0)
ZERO_COST = BacktestConfig(fee_bps=0.0, slippage_bps=0.0)

GRID: list[dict[str, int]] = [
    {"fast": 5, "slow": 20},
    {"fast": 10, "slow": 30},
    {"fast": 20, "slow": 60},
]

# the two-regime fixture's grid: one fast pair, one slow pair (see regime_shift_prices)
MOBILE_GRID: list[dict[str, int]] = [
    {"fast": 3, "slow": 10},
    {"fast": 10, "slow": 40},
]


def _mobile_fixture() -> tuple[pd.DataFrame, list[TemporalSplit]]:
    """Prices + ROLLING folds calibrated so the argmax flips between folds.

    Rolling (anchored=False) on purpose: the train window slides out of the zigzag
    regime into the cycle regime, which is exactly the configuration where the
    selection can move (an anchored train would dilute the regime change forever).
    purge=40 covers the grid's longest look-back (slow=40).
    """
    prices = regime_shift_prices(seed=7)
    splits = walk_forward_splits(
        len(prices), n_folds=4, test_size=0.4, purge=40, embargo=2, anchored=False
    )
    return prices, splits


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

    ANTI-VACUITY GUARD, asserted first: on this fixture the selection must be NON
    TRIVIAL -- the K folds must retain at least TWO distinct parameter sets. A
    constant selection could never change under perturbation, and the invariant
    would be green by vacuity; a test that cannot fail must declare itself useless.
    """
    prices, splits = _mobile_fixture()
    base = walk_forward_evaluate(prices, _ma_factory, MOBILE_GRID, splits, COSTS)

    picks = {tuple(sorted(f.params.items())) for f in base.folds}
    assert len(picks) >= 2, "fixture dégénérée : la cécité au futur n'est pas testée"

    for k, split in enumerate(splits):
        tampered = prices.copy()
        cutoff = int(split.train[-1]) + 1
        tampered.iloc[cutoff:] = tampered.iloc[cutoff:] * 1.5
        result = walk_forward_evaluate(tampered, _ma_factory, MOBILE_GRID, splits, COSTS)
        for j in range(k + 1):
            assert result.folds[j].params == base.folds[j].params
            assert result.folds[j].train_metric == base.folds[j].train_metric


def test_fold_test_returns_ignore_future_bars() -> None:
    """Truncating the series right after fold k's test window changes NOTHING in
    folds < k (bit-for-bit) and nothing in fold k EXCEPT the marking of its final
    bar -- treated explicitly below, never waved through with a tolerance.

    The exemption is the documented splice rule, not leakage: an interior fold's
    last bar is marked at the next fold's first open (the honest mark of a position
    held across the boundary); truncation removes that next open, so the truncated
    run marks the bar at its own close. Positions and fees are bit-for-bit
    identical either way, and the truncated last-bar return is pinned analytically
    against the close-marking formula.

    ANTI-VACUITY GUARD: the exemption is only exercised if some interior fold
    actually CARRIES a position at its last bar -- with pos = 0 the close-vs-next-
    open re-marking is invisible (0 == 0 under any marking) and a marking
    regression would pass. seed=2 was scanned so every fold ends with pos = 1.0."""
    prices = random_walk_prices(400, seed=2)
    splits = walk_forward_splits(400, n_folds=3, test_size=0.3, purge=60, embargo=2)
    full = walk_forward_evaluate(prices, _ma_factory, GRID, splits, COSTS)
    assert any(
        float(full.folds[k].test_result.positions.iloc[-1]) != 0.0 for k in range(len(splits) - 1)
    ), "fixture dégénérée : l'exemption de marking de la dernière barre n'est pas testée"
    for k, split in enumerate(splits):
        truncated = prices.iloc[: int(split.test[-1]) + 1]
        result = walk_forward_evaluate(truncated, _ma_factory, GRID, list(splits[: k + 1]), COSTS)
        for j in range(k + 1):
            assert result.folds[j].params == full.folds[j].params
            assert result.folds[j].train_metric == full.folds[j].train_metric
        for j in range(k):  # folds strictly before k: their next bar exists in both
            pd.testing.assert_series_equal(  # exact: no tolerance
                result.folds[j].test_result.returns,
                full.folds[j].test_result.returns,
                check_exact=True,
            )
        r_full = full.folds[k].test_result
        r_trunc = result.folds[k].test_result
        pd.testing.assert_series_equal(  # every bar but the last: bit-for-bit
            r_trunc.returns.iloc[:-1], r_full.returns.iloc[:-1], check_exact=True
        )
        pd.testing.assert_series_equal(r_trunc.positions, r_full.positions, check_exact=True)
        pd.testing.assert_series_equal(r_trunc.turnover, r_full.turnover, check_exact=True)
        # the last bar of the truncated run is marked at its own close: pin it
        # against the engine's formula r = (1 + pos * (close/open - 1)) * (1 - cost) - 1
        t_last = int(split.test[-1])
        pos = float(r_trunc.positions.iloc[-1])
        cost = float(r_trunc.turnover.iloc[-1]) * COSTS.cost_rate
        bar_return = float(prices["close"].iloc[t_last] / prices["open"].iloc[t_last]) - 1.0
        expected_last = (1.0 + pos * bar_return) * (1.0 - cost) - 1.0
        assert abs(float(r_trunc.returns.iloc[-1]) - expected_last) < 1e-15


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


def test_gapped_test_window_is_rejected() -> None:
    """A hand-built split whose TEST window skips bars must be rejected loudly: the
    engine's next-open marking would compress the whole gap move into the pre-gap
    bar and bill it to the strategy as P&L -- silently wrong returns, never
    accepted. (assert_no_leakage alone passes such a window: strictly increasing,
    disjoint, gapped from train -- the contiguity check is a separate guard.)"""
    prices = random_walk_prices(400, seed=3)
    good = walk_forward_splits(400, n_folds=1, test_size=0.3, purge=60, embargo=2)[0]
    bad = TemporalSplit(
        train=good.train,
        test=np.concatenate([good.test[:20], good.test[40:]]),
        purge=good.purge,
        embargo=good.embargo,
    )
    with pytest.raises(ValueError, match="not contiguous"):
        walk_forward_evaluate(prices, _ma_factory, GRID, [bad], COSTS)


def test_unknown_calendar_hole_is_rejected_loudly_and_listed() -> None:
    """A bar missing from the CALENDAR (a data hole) that is NOT whitelisted must
    fail loudly and be listed by timestamp -- the walk-forward validator applies
    the same hole discipline as the data loader. (Positional indices stay
    contiguous when a row is absent from the frame, so the positional check alone
    cannot see this; the calendar check must.)"""
    full = make_prices((100.0 * np.cumprod(1.0 + np.full(400, 0.003))).tolist())
    dropped_ts = full.index[350]
    holey = full.drop(index=dropped_ts)
    splits = walk_forward_splits(len(holey), n_folds=1, test_size=0.2, purge=30, embargo=0)
    with pytest.raises(ValueError, match="not whitelisted") as excinfo:
        walk_forward_evaluate(holey, _ma_factory, [{"fast": 5, "slow": 20}], splits, COSTS)
    assert dropped_ts.isoformat() in str(excinfo.value)  # the hole is LISTED


def test_whitelisted_hole_is_accepted_and_stays_a_hole() -> None:
    """A hole in the whitelist (the manifest's pinned missing_bars) is accepted:
    the run completes, NO bar is invented at the hole's timestamp, and the position
    is simply held across it (the pre-hole bar marks at the next available open --
    real market movement, not an invented return)."""
    full = make_prices((100.0 * np.cumprod(1.0 + np.full(400, 0.003))).tolist())
    dropped_ts = full.index[350]
    holey = full.drop(index=dropped_ts)
    splits = walk_forward_splits(len(holey), n_folds=1, test_size=0.2, purge=30, embargo=0)
    result = walk_forward_evaluate(
        holey,
        _ma_factory,
        [{"fast": 5, "slow": 20}],
        splits,
        COSTS,
        known_holes=pd.DatetimeIndex([dropped_ts]),
    )
    assert dropped_ts not in result.oos.returns.index  # still a hole: nothing invented
    assert len(result.oos.returns) == len(splits[0].test)
    hole_side = int(
        result.oos.positions.index.get_indexer(pd.DatetimeIndex([dropped_ts]), method="ffill")[0]
    )
    assert float(result.oos.positions.iloc[hole_side]) == 1.0  # held across the hole


def test_hole_whitelist_mechanism_leaves_gap_free_results_bit_identical() -> None:
    """On gap-free data the whitelist mechanism must change NOTHING, bit-for-bit:
    no whitelist, an empty whitelist, and an unused whitelist entry all produce
    identical results. (On the real gap-free 1d data the same guarantee is enforced
    byte-for-byte by test_metrics_golden.py.)"""
    prices = random_walk_prices(400, seed=13)
    splits = walk_forward_splits(400, n_folds=4, test_size=0.3, purge=30, embargo=2)
    base = walk_forward_evaluate(prices, _ma_factory, GRID, splits, COSTS)
    for whitelist in (
        pd.DatetimeIndex([]),
        pd.DatetimeIndex([prices.index[0] - pd.Timedelta(days=30)]),  # unused entry
    ):
        other = walk_forward_evaluate(
            prices, _ma_factory, GRID, splits, COSTS, known_holes=whitelist
        )
        pd.testing.assert_series_equal(other.oos.returns, base.oos.returns, check_exact=True)
        assert other.metrics == base.metrics
        assert [f.params for f in other.folds] == [f.params for f in base.folds]


def test_gapped_train_window_is_rejected() -> None:
    """Same guard for the TRAIN window: a gapped train would silently misprice
    every selection candidate through the same compressed-gap marking."""
    prices = random_walk_prices(400, seed=3)
    good = walk_forward_splits(400, n_folds=1, test_size=0.3, purge=60, embargo=2)[0]
    bad = TemporalSplit(
        train=np.concatenate([good.train[:50], good.train[100:]]),
        test=good.test,
        purge=good.purge,
        embargo=good.embargo,
    )
    with pytest.raises(ValueError, match="not contiguous"):
        walk_forward_evaluate(prices, _ma_factory, GRID, [bad], COSTS)


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

    # OOS equity compounds the concatenated returns (fold-local equity restarts
    # at 1.0 for reporting, but the OOS curve is one continuous compounding)
    pd.testing.assert_series_equal(
        result.oos.equity, (1.0 + expected).cumprod(), check_exact=True, check_names=False
    )


# --------------------------------------------------------------------------- #
# the fold boundary: same params splice exactly, a param change resets to flat
# --------------------------------------------------------------------------- #


def test_adjacent_folds_splice_exactly() -> None:
    """If consecutive folds retain the SAME parameters, their concatenated OOS
    returns must equal, bar for bar, ONE backtest run over the union of their test
    windows: no artificial flat first bar, no phantom re-entry fee at any fold
    boundary. (A single-candidate grid forces identical parameters on every fold,
    so the whole OOS region must splice into one continuous run.)"""
    prices = random_walk_prices(400, seed=21)
    splits = walk_forward_splits(400, n_folds=4, test_size=0.3, purge=60, embargo=2)
    single = [{"fast": 5, "slow": 20}]
    result = walk_forward_evaluate(prices, _ma_factory, single, splits, COSTS)

    union = np.concatenate([np.asarray(s.test) for s in splits])
    history = prices.iloc[: int(union[-1]) + 1]
    signal = _ma_factory(history, single[0])
    direct = run_backtest(prices.iloc[union], signal.iloc[union], COSTS)

    np.testing.assert_allclose(  # spec tolerance 1e-12 (in practice bit-for-bit)
        result.oos.returns.to_numpy(), direct.returns.to_numpy(), rtol=0.0, atol=1e-12
    )
    pd.testing.assert_series_equal(
        result.oos.positions, direct.positions, check_exact=True, check_names=False
    )
    pd.testing.assert_series_equal(
        result.oos.turnover, direct.turnover, check_exact=True, check_names=False
    )


def test_param_change_resets_position_to_flat_between_folds() -> None:
    """A REAL parameter change between folds resets to flat: re-allocating to a new
    parameterization is actual behavior, not a splice artifact. Uses the two-regime
    fixture, whose selection provably changes between folds (anti-vacuity below)."""
    prices, splits = _mobile_fixture()
    result = walk_forward_evaluate(prices, _ma_factory, MOBILE_GRID, splits, COSTS)
    changes = [
        k
        for k in range(1, len(result.folds))
        if result.folds[k].params != result.folds[k - 1].params
    ]
    assert changes, "fixture dégénérée : aucun changement de paramètres à tester"
    for k in changes:
        assert float(result.folds[k].test_result.positions.iloc[0]) == 0.0


def test_non_adjacent_same_param_folds_do_not_splice() -> None:
    """Two folds may retain the SAME parameters and still be separated by a time
    gap (hand-built splits): the splice rule must NOT glue them. Gluing would carry
    a position across bars the evaluation excludes and mark the pre-gap bar at an
    open beyond the gap -- silent mispricing. The second fold must start flat, its
    returns must equal a standalone backtest of its own window, and no boundary may
    count as carried. (Born of mutation M7, which survived the previous suite.)"""
    prices = make_prices((100.0 * np.cumprod(1.0 + np.full(400, 0.003))).tolist())
    single = [{"fast": 5, "slow": 20}]
    fold_a = TemporalSplit(
        train=np.arange(0, 200, dtype=np.int64),
        test=np.arange(230, 270, dtype=np.int64),
        purge=30,
        embargo=0,
    )
    fold_b = TemporalSplit(
        train=np.arange(0, 260, dtype=np.int64),
        test=np.arange(290, 330, dtype=np.int64),
        purge=30,
        embargo=0,
    )
    result = walk_forward_evaluate(prices, _ma_factory, single, [fold_a, fold_b], COSTS)
    assert result.folds[0].params == result.folds[1].params  # same params, on purpose

    history = prices.iloc[:330]
    signal = _ma_factory(history, single[0])
    # anti-vacuity: the signal IS long at the pre-gap boundary bar, so an incorrect
    # glue would necessarily carry a non-zero position -- this test cannot pass
    # vacuously the way an all-flat boundary would let it
    assert float(signal.iloc[269]) == 1.0

    assert result.n_carried_boundaries == 0
    assert float(result.folds[1].test_result.positions.iloc[0]) == 0.0  # no carry over the gap
    standalone = run_backtest(prices.iloc[290:330], signal.iloc[290:330], COSTS)
    pd.testing.assert_series_equal(
        result.folds[1].test_result.returns, standalone.returns, check_exact=True
    )


def test_n_carried_boundaries_is_counted_by_the_engine() -> None:
    """The number that moved the OOS figures when splicing was introduced is a
    first-class, engine-counted output -- never inferred by hand.

    Cross-checked two ways: against the per-fold position slices (catches offset /
    bookkeeping bugs in the segment slicing), and on the two-regime fixture where a
    param-change boundary must never count as carried."""
    prices = random_walk_prices(400, seed=21)
    splits = walk_forward_splits(400, n_folds=4, test_size=0.3, purge=60, embargo=2)
    result = walk_forward_evaluate(prices, _ma_factory, [{"fast": 5, "slow": 20}], splits, COSTS)
    from_slices = sum(
        1
        for k in range(1, len(result.folds))
        if float(result.folds[k].test_result.positions.iloc[0]) != 0.0
    )
    assert result.n_carried_boundaries == from_slices
    assert result.n_carried_boundaries > 0, (
        "fixture dégénérée : aucune frontière portante, le compteur n'est pas testé"
    )

    prices2, splits2 = _mobile_fixture()
    result2 = walk_forward_evaluate(prices2, _ma_factory, MOBILE_GRID, splits2, COSTS)
    carried_at_same_param_boundaries = sum(
        1
        for k in range(1, len(result2.folds))
        if result2.folds[k].params == result2.folds[k - 1].params
        and float(result2.folds[k].test_result.positions.iloc[0]) != 0.0
    )
    assert result2.n_carried_boundaries == carried_at_same_param_boundaries


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

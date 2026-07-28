"""Oracle 4: full-engine cross-check against an independent implementation (vectorbt).

Same real BTC data, same strategies (buy & hold, MA 20/50 long-only), same execution
timing (signal at close[t], fill at open[t+1]) and same fee rate on both engines.

Convention alignment (controlled variables, not concessions):

- Execution: our engine turns signal[t] into a fill at open[t+1]. vectorbt executes an
  entry/exit on its own bar at ``price``; we therefore hand vectorbt the engine's
  position series (signal shifted by one bar) as entries/exits, with price = open.
- Marking: our engine marks equity at the NEXT open (final bar at its own close);
  vectorbt marks at its ``close`` series. We hand vectorbt ``open.shift(-1)`` (last
  bar: close) as its "close", so both engines value the portfolio identically.

With fees = 0 the two engines must then coincide to float precision.

The ONE remaining difference with fees > 0 is the fee model, isolated exactly:

- ours: a buy multiplies equity by (1 - c)  (fee charged on traded notional),
- vectorbt: buying with all cash solves size * price * (1 + c) = cash, factor 1/(1+c).

Per buy the ratio is (1 - c)(1 + c) = 1 - c^2, hence the exact accounting identity

    1 + total_return_ours = (1 + total_return_vbt) * (1 - c^2) ** n_buys

asserted at 1e-9 relative -- an identity, never a loosened tolerance. With c = 10 bps,
c^2 = 1e-6 per buy.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
import pytest
import vectorbt as vbt

from data.loader import load_ohlcv
from engine.backtest import BacktestConfig, BacktestResult, run_backtest
from engine.metrics import max_drawdown, sharpe_ratio, total_return
from strategies import buy_and_hold_signal, ma_crossover_signal

# pin vectorbt's annualization to the engine's A = 365 (its default, made explicit)
vbt.settings.returns["year_freq"] = "365 days"

Strategy = Callable[[pd.DataFrame], pd.Series]

STRATEGIES: list[tuple[str, Strategy]] = [
    ("buy_and_hold", buy_and_hold_signal),
    ("ma_crossover_long_only", ma_crossover_signal),
]


@pytest.fixture(scope="module", name="btc")
def btc_fixture() -> pd.DataFrame:
    return load_ohlcv("BTC/USDT")


def _run_both(
    prices: pd.DataFrame, signal: pd.Series, fee_bps: float
) -> tuple[BacktestConfig, BacktestResult, Any, int]:
    cfg = BacktestConfig(fee_bps=fee_bps, slippage_bps=0.0)
    ours = run_backtest(prices, signal, cfg)
    pos = signal.shift(1).fillna(0.0)
    delta = pos.diff().fillna(pos)
    mark = prices["open"].shift(-1).fillna(prices["close"])
    pf = vbt.Portfolio.from_signals(
        close=mark,
        entries=delta > 0.0,
        exits=delta < 0.0,
        price=prices["open"],
        fees=cfg.cost_rate,
        init_cash=1.0,
        size=np.inf,
        freq="1D",
    )
    return cfg, ours, pf, int((delta > 0.0).sum())


@pytest.mark.parametrize(("name", "strategy"), STRATEGIES)
def test_zero_fee_engines_coincide_to_float_precision(
    btc: pd.DataFrame, name: str, strategy: Strategy
) -> None:
    _, ours, pf, _ = _run_both(btc, strategy(btc), fee_bps=0.0)
    np.testing.assert_allclose(
        ours.returns.to_numpy(), pf.returns().to_numpy(), rtol=0.0, atol=1e-12
    )
    assert total_return(ours.returns) == pytest.approx(float(pf.total_return()), rel=1e-9)
    assert sharpe_ratio(ours.returns) == pytest.approx(float(pf.sharpe_ratio()), rel=1e-9)
    assert max_drawdown(ours.returns) == pytest.approx(float(pf.max_drawdown()), rel=1e-9)


@pytest.mark.parametrize(("name", "strategy"), STRATEGIES)
def test_standard_fees_agree_up_to_isolated_fee_model_difference(
    btc: pd.DataFrame, name: str, strategy: Strategy
) -> None:
    cfg, ours, pf, n_buys = _run_both(btc, strategy(btc), fee_bps=10.0)
    c = cfg.cost_rate

    # exact accounting identity for the fee-model difference (see module docstring)
    ours_growth = 1.0 + total_return(ours.returns)
    vbt_growth = 1.0 + float(pf.total_return())
    assert ours_growth == pytest.approx(vbt_growth * (1.0 - c**2) ** n_buys, rel=1e-9)

    # the c^2-per-buy perturbation moves per-bar returns by ~1e-6 on n_buys of the
    # n = 3240 bars, shifting Sharpe and max drawdown by O(n_buys * c^2); with
    # n_buys ~ 40 on this span that is ~4e-5, so the unchanged 1e-4 tolerance keeps
    # a ~2.5x analytic buffer (down from 10x on the 1096-bar sample). Pre-engaged
    # rule (B4): if the observed gap ever exceeds 1e-4, the bound is recomputed
    # from the REAL n_buys and the tolerance becomes that bound x the original 10x
    # margin, derived here -- never widened on sight; an unexplained disagreement
    # with the oracle is a RESULT, not a nuisance.
    assert sharpe_ratio(ours.returns) == pytest.approx(float(pf.sharpe_ratio()), abs=1e-4)
    assert max_drawdown(ours.returns) == pytest.approx(float(pf.max_drawdown()), abs=1e-4)


@pytest.mark.parametrize("fee_bps", [0.0, 10.0])
@pytest.mark.parametrize(("name", "strategy"), STRATEGIES)
def test_trade_count_matches_engine_turnover(
    btc: pd.DataFrame, name: str, strategy: Strategy, fee_bps: float
) -> None:
    # long/flat with unit position: every |delta pos| = 1 is one full-size order, so
    # the engine's summed turnover must equal vectorbt's order count exactly
    _, ours, pf, _ = _run_both(btc, strategy(btc), fee_bps=fee_bps)
    assert int(round(float(ours.turnover.sum()))) == int(pf.orders.count())

"""Vectorized daily backtest with zero look-ahead by construction.

Timing convention (the single most important contract in this repo):

- ``signal[t]`` is the target position decided using information available up to and
  including ``close[t]``.
- The trade is executed at ``open[t+1]``: internally ``position = signal.shift(1)``,
  so ``position[t]`` is the exposure held from ``open[t]`` to the next execution price.
- The portfolio is marked to market at each next open; the final bar is marked at its
  close (there is no next open).
- Window return: ``r[t] = mark[t] / open[t] - 1`` with ``mark[t] = open[t+1]``
  (last bar: ``close[t]``).

Costs: ``fee_bps + slippage_bps`` applied to traded notional ``|Δposition|`` at each
execution, as a multiplicative equity hit:
``equity[t] = equity[t-1] * (1 - |Δpos[t]| * cost_rate) * (1 + pos[t] * r[t])``.

Position sizing: the signal IS the target weight of equity (1.0 = fully long,
0.0 = flat, -1.0 = fully short). No leverage cap is enforced beyond the signal itself.

Guardrail: with ``signal ≡ 1`` this engine must reproduce the asset's gross return net
of the single entry fee EXACTLY (telescoping product) -- enforced by
``tests/test_buy_and_hold_identity.py``.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class BacktestConfig:
    fee_bps: float = 10.0
    slippage_bps: float = 0.0

    @property
    def cost_rate(self) -> float:
        """Total per-trade cost as a fraction of traded notional."""
        return (self.fee_bps + self.slippage_bps) / 10_000.0


@dataclass(frozen=True)
class BacktestResult:
    equity: pd.Series
    returns: pd.Series
    positions: pd.Series
    turnover: pd.Series


def _validate_inputs(prices: pd.DataFrame, signal: pd.Series) -> None:
    missing = {"open", "close"} - set(prices.columns)
    if missing:
        raise ValueError(f"prices is missing columns: {sorted(missing)}")
    if len(prices) < 2:
        raise ValueError("need at least 2 bars to backtest")
    if not prices.index.is_monotonic_increasing:
        raise ValueError("prices index must be sorted in time")
    if prices.index.has_duplicates:
        raise ValueError("prices index has duplicate timestamps")
    if prices[["open", "close"]].isna().any().any():
        raise ValueError("NaN in open/close prices")
    if not signal.index.equals(prices.index):
        raise ValueError("signal index must be identical to prices index")
    if signal.isna().any():
        raise ValueError("NaN in signal -- fill warmup with 0.0, do not drop bars")


def run_backtest(
    prices: pd.DataFrame,
    signal: pd.Series,
    config: BacktestConfig | None = None,
) -> BacktestResult:
    """Run the backtest. See module docstring for the exact timing/cost conventions."""
    cfg = config if config is not None else BacktestConfig()
    _validate_inputs(prices, signal)

    open_ = prices["open"].astype("float64")
    close = prices["close"].astype("float64")

    # signal decided at close[t] -> position in effect from open[t+1]
    positions = signal.astype("float64").shift(1).fillna(0.0)

    # mark[t] = open[t+1], except the last bar which is marked at its own close
    mark = open_.shift(-1).fillna(close)
    window_return = mark / open_ - 1.0

    gross = positions * window_return
    turnover = positions.diff().abs().fillna(positions.abs())
    net = (1.0 + gross) * (1.0 - turnover * cfg.cost_rate) - 1.0
    equity = (1.0 + net).cumprod()

    return BacktestResult(
        equity=equity.rename("equity"),
        returns=net.rename("returns"),
        positions=positions.rename("positions"),
        turnover=turnover.rename("turnover"),
    )

"""Oracle 3: distributional recovery of a KNOWN Sharpe and volatility.

Feed N i.i.d. Normal(m, s) per-bar returns (fixed seed, known ground truth) through the
FULL engine at constant full position with zero costs, then check that:

1. the engine's net returns recover the input returns exactly (timing/passthrough), and
2. the measured annualized Sharpe and volatility recover (m/s)*sqrt(A) and s*sqrt(A)
   within a tolerance DERIVED from the sampling error of the estimators (Lo 2002 for
   the Sharpe, sqrt(2n) for the volatility) -- never a hand-tuned tolerance.

This proves the Sharpe computation and the sqrt(365) annualization are correct in
distribution, complementing the exact hand-computed unit tests in test_metrics.py.
"""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest

from engine.backtest import BacktestConfig, run_backtest
from engine.metrics import sharpe_ratio
from tests.utils import make_prices

N = 200_000
A = 365  # the engine's annualization factor for the default "1d" timeframe
SEED = 20260717

# (mean, std) per bar, chosen so 1 + r stays positive and cumprod stays in float range
CASES = [
    (0.001, 0.02),  # true SR_ann ~ +0.955
    (0.002, 0.025),  # true SR_ann ~ +1.528
    (-0.0005, 0.015),  # true SR_ann ~ -0.637
]


def _prices_from_returns(r: npt.NDArray[np.float64]) -> pd.DataFrame:
    levels = 100.0 * np.concatenate([[1.0], np.cumprod(1.0 + r)])
    # hourly index only to stay inside datetime64[ns] bounds at N=200k bars; the
    # metrics annualize per the timeframe ARGUMENT (default "1d" -> A=365), never
    # by reading the index frequency
    return make_prices(levels, freq="h")


@pytest.mark.parametrize(("m", "s"), CASES)
def test_engine_recovers_known_sharpe_and_vol(m: float, s: float) -> None:
    rng = np.random.default_rng(SEED)
    r = rng.normal(m, s, size=N)
    assert (1.0 + r > 0.0).all()

    prices = _prices_from_returns(r)
    signal = pd.Series(1.0, index=prices.index)
    result = run_backtest(prices, signal, BacktestConfig(fee_bps=0.0, slippage_bps=0.0))

    # bar 0 is still flat (signal executes at open[1]); the last bar is marked at its
    # own close (no next open) and contributes a 0 return by construction
    active = result.returns.to_numpy()[1:-1]
    np.testing.assert_allclose(active, r[1:], rtol=1e-9, atol=1e-12)

    n = active.size
    measured_sharpe = sharpe_ratio(pd.Series(active))  # default "1d" -> sqrt(A)
    measured_vol = float(np.std(active, ddof=1) * math.sqrt(A))
    true_sharpe = m / s * math.sqrt(A)
    true_vol = s * math.sqrt(A)

    # SE(SR_bar) ~ sqrt((1 + SR_bar^2 / 2) / n)  (Lo 2002), annualized by sqrt(A);
    # SE(vol) ~ vol / sqrt(2n). 4 standard errors: p(false alarm) ~ 6e-5 per case.
    se_sharpe = math.sqrt((1.0 + 0.5 * (m / s) ** 2) / n) * math.sqrt(A)
    se_vol = true_vol / math.sqrt(2.0 * n)
    assert abs(measured_sharpe - true_sharpe) <= 4.0 * se_sharpe
    assert abs(measured_vol - true_vol) <= 4.0 * se_vol

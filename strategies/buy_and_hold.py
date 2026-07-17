"""Buy & hold: target position 1.0 at every close.

With the engine's timing convention the first fill happens at open[1]; the engine must
then reproduce the asset's gross return net of the single entry fee exactly (this is
the anti-bug guardrail tested in tests/test_buy_and_hold_identity.py).
"""

from __future__ import annotations

import pandas as pd


def buy_and_hold_signal(prices: pd.DataFrame) -> pd.Series:
    return pd.Series(1.0, index=prices.index, name="signal")

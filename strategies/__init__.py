"""Baseline strategies. A strategy is a function: prices DataFrame -> signal Series.

The signal at index t may only use information available up to and including close[t]
(causality is enforced by tests/test_strategies.py).
"""

from strategies.buy_and_hold import buy_and_hold_signal
from strategies.ma_crossover import ma_crossover_signal

__all__ = ["buy_and_hold_signal", "ma_crossover_signal"]

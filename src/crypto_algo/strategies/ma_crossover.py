"""Moving average crossover — baseline strategy.

Hypothesis: when a fast MA is above a slow MA, the asset is trending up;
hold it. Otherwise, hold cash.

Signal at bar i uses data through bar i (close known). We shift by 1 so the
decision executes at the open of bar i+1. Strictly causal.

This is a BASELINE. Its purpose is to exercise the pipeline.
"""

from __future__ import annotations

import pandas as pd

from crypto_algo.features.engine import moving_average


def ma_crossover_target(
    close: pd.Series,
    fast_window: int,
    slow_window: int,
) -> pd.Series:
    if fast_window >= slow_window:
        raise ValueError(
            f"fast_window ({fast_window}) must be < slow_window ({slow_window})"
        )

    fast = moving_average(close, window=fast_window)
    slow = moving_average(close, window=slow_window)
    signal = (fast > slow).astype(float)
    target = signal.shift(1)
    target.iloc[0] = 0.0
    return target
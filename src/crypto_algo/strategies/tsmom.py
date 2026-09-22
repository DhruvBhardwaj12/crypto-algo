"""Time-Series Momentum (TSMOM).

Rule:
  At bar close, look back N bars. If the cumulative return over that
  window is positive, target = 1 (long). Otherwise target = 0 (flat).

We shift the signal by 1 bar so execution happens at the next bar's open,
matching our simulator's convention. The signal is strictly causal.
"""

from __future__ import annotations

import pandas as pd


def tsmom_target(close: pd.Series, lookback_n: int = 42) -> pd.Series:
    """Return target position series (0 or 1), aligned to close.index."""
    if lookback_n <= 0:
        raise ValueError(f"lookback_n must be positive, got {lookback_n}")

    # Trailing log return over lookback_n bars. Uses only past + current
    # close, no future. First valid value at index = lookback_n.
    log_ret = (close / close.shift(lookback_n)).apply(lambda x: x)
    signal = (log_ret > 1.0).astype(float)  # ratio > 1 means price rose
    signal = signal.fillna(0.0)

    # Execute next bar: shift the signal forward by 1.
    target = signal.shift(1).fillna(0.0)
    return target

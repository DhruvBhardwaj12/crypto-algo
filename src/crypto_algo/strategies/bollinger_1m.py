"""1-minute Bollinger Band mean reversion.

At 1m bar close:
  - Compute 20-bar SMA and 20-bar std of close
  - Lower band = SMA - 2 * std
  - Entry: close < lower band → target = 1
  - Exit:  close > SMA          → target = 0

Signal used at next bar open. Long-only, unlevered.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def bollinger_band_target(
    df: pd.DataFrame,
    window: int = 20,
    num_std: float = 2.0,
) -> pd.Series:
    """Return target position (0 or 1) aligned to df.index."""
    if window <= 1:
        raise ValueError(f"window must be > 1, got {window}")

    close = df["close"].reset_index(drop=True)
    sma = close.rolling(window, min_periods=window).mean()
    std = close.rolling(window, min_periods=window).std()
    lower = sma - num_std * std

    n = len(df)
    target = np.zeros(n, dtype=float)
    in_pos = False

    close_arr = close.to_numpy()
    sma_arr = sma.to_numpy()
    lower_arr = lower.to_numpy()

    for i in range(n):
        if np.isnan(sma_arr[i]) or np.isnan(lower_arr[i]):
            target[i] = 1.0 if in_pos else 0.0
            continue

        if not in_pos:
            if close_arr[i] < lower_arr[i]:
                in_pos = True
        else:
            if close_arr[i] > sma_arr[i]:
                in_pos = False

        target[i] = 1.0 if in_pos else 0.0

    return pd.Series(target, index=df.index, name="target")

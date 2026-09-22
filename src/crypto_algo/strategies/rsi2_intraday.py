"""RSI(2) mean reversion on 5m bars.

At 5m bar close:
  - Compute RSI(2) on close
  - Entry: RSI(2) < 10 → target = 1
  - Exit:  RSI(2) > 70 → target = 0

Signal used at next 5m bar open. Long-only, unlevered.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def rsi_wilder(close: pd.Series, period: int = 2) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100.0 - 100.0 / (1.0 + rs)


def rsi2_intraday_target(
    df: pd.DataFrame,
    rsi_period: int = 2,
    entry_threshold: float = 10.0,
    exit_threshold: float = 70.0,
) -> pd.Series:
    """Return target position (0 or 1) aligned to df.index."""
    close = df["close"].reset_index(drop=True)
    rsi = rsi_wilder(close, period=rsi_period)

    n = len(df)
    target = np.zeros(n, dtype=float)
    in_pos = False
    rsi_arr = rsi.to_numpy()

    for i in range(n):
        if np.isnan(rsi_arr[i]):
            target[i] = 1.0 if in_pos else 0.0
            continue

        if not in_pos:
            if rsi_arr[i] < entry_threshold:
                in_pos = True
        else:
            if rsi_arr[i] > exit_threshold:
                in_pos = False

        target[i] = 1.0 if in_pos else 0.0

    return pd.Series(target, index=df.index, name="target")

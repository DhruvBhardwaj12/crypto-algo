"""Leveraged Trend Rider.

Long when price is above MA(50) and RSI(14) > 45.
Exit when price crosses back below MA(50).

Signal is 0 or 1. Position size = signal × leverage.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def rsi_wilder(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100.0 - 100.0 / (1.0 + rs)


def leveraged_trend_signal(
    df: pd.DataFrame,
    ma_window: int = 50,
    rsi_window: int = 14,
    rsi_entry: float = 45.0,
) -> pd.Series:
    """Return target signal (0 or 1) aligned to df.index. Shifted for next-bar exec."""
    close = df["close"].reset_index(drop=True)
    ma = close.rolling(ma_window, min_periods=ma_window).mean()
    rsi = rsi_wilder(close, rsi_window)

    n = len(close)
    target = np.zeros(n, dtype=float)
    in_pos = False
    close_arr = close.to_numpy()
    ma_arr = ma.to_numpy()
    rsi_arr = rsi.to_numpy()

    for i in range(n):
        if np.isnan(ma_arr[i]) or np.isnan(rsi_arr[i]):
            target[i] = 1.0 if in_pos else 0.0
            continue

        if not in_pos:
            if close_arr[i] > ma_arr[i] and rsi_arr[i] > rsi_entry:
                in_pos = True
        else:
            if close_arr[i] < ma_arr[i]:
                in_pos = False

        target[i] = 1.0 if in_pos else 0.0

    signal = pd.Series(target, index=df.index, name="target")
    return signal.shift(1).fillna(0.0)

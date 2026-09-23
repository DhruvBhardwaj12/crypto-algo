"""Funding carry signal v2.

Enter when smoothed funding is above a threshold (positive yield to shorts).
Exit when smoothed funding drops below zero.

Smoothing uses a rolling mean of the last 9 funding events (3 days).
Signal lags by 1 event to avoid look-ahead.
"""

from __future__ import annotations

import pandas as pd


def funding_carry_signal_v2(
    funding_rate: pd.Series,
    smoothing: int = 9,
    entry_threshold: float = 0.00005,  # 0.5 bps per 8h ~ 5.5% annualized
    exit_threshold: float = 0.0,
) -> pd.Series:
    """Return 0/1 signal aligned to funding_rate index. Lagged by 1 event."""
    smoothed = funding_rate.rolling(smoothing, min_periods=smoothing).mean()
    smoothed_lagged = smoothed.shift(1)

    signal = pd.Series(0, index=funding_rate.index, dtype=int)
    in_pos = False
    for i in range(len(funding_rate)):
        s = smoothed_lagged.iloc[i]
        if pd.isna(s):
            signal.iloc[i] = 0
            continue
        if not in_pos and s > entry_threshold:
            in_pos = True
        elif in_pos and s < exit_threshold:
            in_pos = False
        signal.iloc[i] = 1 if in_pos else 0
    return signal

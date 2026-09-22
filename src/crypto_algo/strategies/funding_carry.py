"""Funding carry entry/exit rules.

Hypothesis: positive funding is a persistent condition. When funding
over a recent window is positive, shorts are being paid to hold. We enter
a delta-neutral carry position and hold it while funding remains positive
above a threshold.

Exit when funding falls below the threshold or turns negative.

We deliberately keep this rule simple. Complexity belongs in testing,
not in signals.
"""

from __future__ import annotations

import pandas as pd


def funding_carry_signal(
    funding_rate: pd.Series,
    smoothing_events: int = 9,        # 9 events = 3 days
    entry_threshold_bps: float = 0.5,  # 0.005% per 8h ~ 5.5% annualized
    exit_threshold_bps: float = 0.0,
) -> pd.Series:
    """Return a 0/1 position series aligned to funding events.

    Parameters
    ----------
    funding_rate : per-event funding rate (decimal, e.g., 0.0001)
    smoothing_events : how many funding events to average over
    entry_threshold_bps : enter when smoothed funding > this
    exit_threshold_bps : exit when smoothed funding < this

    Uses only past events (rolling mean, then shift by 1) so the signal
    at event i is fully known before event i occurs.
    """
    if smoothing_events < 1:
        raise ValueError("smoothing_events must be >= 1")

    smoothed = funding_rate.rolling(smoothing_events, min_periods=smoothing_events).mean()
    # Shift by 1 so we act on data from the previous event.
    smoothed_lagged = smoothed.shift(1)

    entry_thresh = entry_threshold_bps / 10_000.0
    exit_thresh = exit_threshold_bps / 10_000.0

    signal = pd.Series(0, index=funding_rate.index, dtype=int)
    in_pos = False
    for i in range(len(funding_rate)):
        s = smoothed_lagged.iloc[i]
        if pd.isna(s):
            signal.iloc[i] = 0
            continue
        if not in_pos and s > entry_thresh:
            in_pos = True
        elif in_pos and s < exit_thresh:
            in_pos = False
        signal.iloc[i] = 1 if in_pos else 0

    return signal

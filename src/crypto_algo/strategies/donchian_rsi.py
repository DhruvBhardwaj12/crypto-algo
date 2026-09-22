"""Multi-timeframe Donchian breakout + RSI regime filter.

Entry (all must be true at 1H bar close):
  - 1H close > prior 55-bar 1H high (Donchian breakout)
  - 1H RSI(14) > 50  (short-term momentum)
  - 4H RSI(14) > 52  (higher-timeframe regime filter)

Exit (any triggers):
  - 1H close < prior 20-bar 1H low  (trailing Donchian exit)
  - 4H RSI(14) < 48  (higher-timeframe momentum failed)

Position is long-only, target in {0, 1}, unlevered.
All signals computed using data through the current bar's close.
Execution is at the next bar's open, handled by the simulator.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's RSI. Causal (EMA-style)."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100.0 - 100.0 / (1.0 + rs)


def donchian_upper(high: pd.Series, lookback: int) -> pd.Series:
    """Prior `lookback`-bar high, shifted 1 (excludes current bar)."""
    return high.rolling(lookback, min_periods=lookback).max().shift(1)


def donchian_lower(low: pd.Series, lookback: int) -> pd.Series:
    """Prior `lookback`-bar low, shifted 1."""
    return low.rolling(lookback, min_periods=lookback).min().shift(1)


def tag_with_4h_regime(
    df_1h: pd.DataFrame,
    df_4h: pd.DataFrame,
    rsi_period: int = 14,
) -> pd.Series:
    """For each 1H bar, return the RSI of the most recent CLOSED 4H bar.

    Uses merge_asof on close_time with allow_exact_matches=False, so the
    4H bar used at 1H bar t is strictly fully closed before t closes.
    """
    df_4h = df_4h.sort_values("close_time").reset_index(drop=True).copy()
    df_4h["rsi_4h"] = rsi(df_4h["close"], rsi_period)

    left = df_1h[["close_time"]].reset_index(drop=True)
    merged = pd.merge_asof(
        left,
        df_4h[["close_time", "rsi_4h"]],
        on="close_time",
        direction="backward",
        allow_exact_matches=False,
    )
    return merged["rsi_4h"].reset_index(drop=True)


def donchian_rsi_target(
    df_1h: pd.DataFrame,
    rsi_4h_series: pd.Series,
    donchian_entry_lookback: int = 55,
    donchian_exit_lookback: int = 20,
    rsi_1h_period: int = 14,
    rsi_1h_entry: float = 50.0,
    rsi_4h_entry: float = 52.0,
    rsi_4h_exit: float = 48.0,
) -> pd.Series:
    """Produce target position (0 or 1) aligned to df_1h index."""
    close = df_1h["close"].reset_index(drop=True)
    high = df_1h["high"].reset_index(drop=True)
    low = df_1h["low"].reset_index(drop=True)
    rsi_4h = rsi_4h_series.reset_index(drop=True)

    rsi_1h = rsi(close, rsi_1h_period)
    dc_up = donchian_upper(high, donchian_entry_lookback)
    dc_dn = donchian_lower(low, donchian_exit_lookback)

    n = len(df_1h)
    target = np.zeros(n, dtype=float)
    in_pos = False

    close_arr = close.to_numpy()
    rsi_1h_arr = rsi_1h.to_numpy()
    rsi_4h_arr = rsi_4h.to_numpy()
    dc_up_arr = dc_up.to_numpy()
    dc_dn_arr = dc_dn.to_numpy()

    for i in range(n):
        # Warmup: keep whatever position we had (flat at the start).
        if (
            np.isnan(rsi_1h_arr[i])
            or np.isnan(rsi_4h_arr[i])
            or np.isnan(dc_up_arr[i])
            or np.isnan(dc_dn_arr[i])
        ):
            target[i] = 1.0 if in_pos else 0.0
            continue

        if not in_pos:
            if (
                close_arr[i] > dc_up_arr[i]
                and rsi_1h_arr[i] > rsi_1h_entry
                and rsi_4h_arr[i] > rsi_4h_entry
            ):
                in_pos = True
        else:
            if (
                close_arr[i] < dc_dn_arr[i]
                or rsi_4h_arr[i] < rsi_4h_exit
            ):
                in_pos = False

        target[i] = 1.0 if in_pos else 0.0

    return pd.Series(target, index=df_1h.index, name="target")

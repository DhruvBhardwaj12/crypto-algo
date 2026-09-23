"""Cross-sectional momentum with trend + regime filters + vol targeting.

Long-only. Once per day:
  1. Check regime: if BTC close <= BTC 200-day MA → flat for the day.
  2. Among symbols whose close > 50-day MA, rank by blended momentum.
  3. Long top-3, each sized by inverse-vol (capped at equal weight).

Signal computed on daily bars (resampled from 4H), then broadcast back to
the 4H grid by forward-fill. The portfolio simulator lags weights by one
bar before execution, so a signal decided at daily close T executes at
the open of the first 4H bar on day T+1. Correct causality.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Fixed parameters — DO NOT TUNE.
MA_TREND_WINDOW = 50        # days
MA_REGIME_WINDOW = 200      # days
MOMENTUM_SHORT = 7          # days
MOMENTUM_LONG = 30          # days
VOL_WINDOW = 20             # days
TARGET_VOL = 0.03           # 3% daily
TOP_N = 3
REGIME_SYMBOL = "BTCUSDT"


def _daily_close(prices_4h: pd.DataFrame) -> pd.DataFrame:
    """Resample 4H closes to daily (last close per UTC day)."""
    daily = prices_4h.resample("1D").last().dropna(how="all")
    return daily


def build_filtered_weights(
    prices_4h: pd.DataFrame,
    top_n: int = TOP_N,
    regime_symbol: str = REGIME_SYMBOL,
) -> pd.DataFrame:
    """Return wide target weights on the 4H grid. Values in [0, 1]."""
    if regime_symbol not in prices_4h.columns:
        raise ValueError(f"Regime symbol {regime_symbol} not in prices.")

    daily = _daily_close(prices_4h)

    # --- Regime check ---
    btc = daily[regime_symbol]
    btc_ma200 = btc.rolling(MA_REGIME_WINDOW, min_periods=MA_REGIME_WINDOW).mean()
    regime_on = (btc > btc_ma200).fillna(False)

    # --- Per-symbol trend filter ---
    ma50 = daily.rolling(MA_TREND_WINDOW, min_periods=MA_TREND_WINDOW).mean()
    above_trend = daily > ma50  # boolean DataFrame

    # --- Momentum score ---
    ret_7 = np.log(daily / daily.shift(MOMENTUM_SHORT))
    ret_30 = np.log(daily / daily.shift(MOMENTUM_LONG))
    score = 0.5 * ret_7 + 0.5 * ret_30

    # --- Realized vol (daily) ---
    daily_ret = np.log(daily).diff()
    realized_vol = daily_ret.rolling(VOL_WINDOW, min_periods=VOL_WINDOW).std()

    # --- Build daily weights ---
    daily_weights = pd.DataFrame(0.0, index=daily.index, columns=daily.columns)

    for t in daily.index:
        # Regime off → flat.
        if not bool(regime_on.loc[t]):
            continue

        # Eligible symbols: above trend, have score, have vol.
        row_score = score.loc[t].dropna()
        row_above = above_trend.loc[t]
        row_vol = realized_vol.loc[t]

        eligible = []
        for sym in row_score.index:
            if sym == regime_symbol:
                pass  # BTC can be traded too
            if not bool(row_above.get(sym, False)):
                continue
            if pd.isna(row_vol.get(sym, np.nan)) or row_vol.loc[sym] <= 0:
                continue
            eligible.append(sym)

        if not eligible:
            continue

        # Rank eligible by score descending, take top N.
        ranked = row_score.loc[eligible].sort_values(ascending=False)
        picks = ranked.head(top_n).index.tolist()

        # Vol-targeted sizing.
        base = 1.0 / top_n
        for sym in picks:
            v = float(row_vol.loc[sym])
            scale = min(1.0, TARGET_VOL / v) if v > 0 else 0.0
            daily_weights.loc[t, sym] = base * scale

    # --- Broadcast daily weights to 4H grid, forward-fill ---
    # Shift by one day so that a weight decided using day T's close is only
    # acted upon from day T+1 onward. This removes the look-ahead bias.
    daily_weights = daily_weights.shift(1)
    weights_4h = daily_weights.reindex(prices_4h.index, method="ffill").fillna(0.0)
    return weights_4h

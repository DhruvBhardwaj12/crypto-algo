"""Causal feature computation.

Every feature at row T uses ONLY data from rows <= T. This is non-negotiable.
Pandas rolling operations are inherently causal (they look backward), so we
use them safely. A separate test verifies this explicitly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def log_returns(close: pd.Series, periods: int = 1) -> pd.Series:
    """Log return over N periods: log(close_t / close_{t-N})."""
    if periods <= 0:
        raise ValueError(f"periods must be positive, got {periods}")
    return np.log(close / close.shift(periods))


def rolling_volatility(returns: pd.Series, window: int) -> pd.Series:
    """Rolling std of returns. Causal: at row T uses rows [T-window+1 .. T]."""
    if window <= 1:
        raise ValueError(f"window must be > 1, got {window}")
    return returns.rolling(window=window, min_periods=window).std()


def moving_average(close: pd.Series, window: int) -> pd.Series:
    """Simple moving average. Causal by construction."""
    if window <= 0:
        raise ValueError(f"window must be positive, got {window}")
    return close.rolling(window=window, min_periods=window).mean()


def zscore(series: pd.Series, window: int) -> pd.Series:
    """Rolling z-score: (x - rolling_mean) / rolling_std. Causal."""
    if window <= 1:
        raise ValueError(f"window must be > 1, got {window}")
    mean = series.rolling(window=window, min_periods=window).mean()
    std = series.rolling(window=window, min_periods=window).std()
    return (series - mean) / std.replace(0, np.nan)


def build_features(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Attach a standard feature set. Original columns are preserved."""
    out = df.copy()
    close = out["close"]
    out["ret_1"] = log_returns(close, periods=1)
    out["ret_24"] = log_returns(close, periods=24)
    out["vol_24"] = rolling_volatility(out["ret_1"], window=24)
    out["vol_168"] = rolling_volatility(out["ret_1"], window=168)
    out["ma_fast"] = moving_average(close, window=config["ma_fast"])
    out["ma_slow"] = moving_average(close, window=config["ma_slow"])
    out["zscore_24"] = zscore(close, window=24)
    return out
    
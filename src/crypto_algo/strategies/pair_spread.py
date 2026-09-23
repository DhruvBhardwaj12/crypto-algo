"""Pair spread mean reversion.

Given two correlated assets (BTC, ETH), compute the ratio. When the
z-score of the ratio diverges beyond a threshold, bet on reversion via
a market-neutral two-leg position.

Signal is a per-symbol weight (BTC weight, ETH weight). The sum is always
near zero (market-neutral). Gross = 1.0 when in position.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def pair_spread_weights(
    btc_close: pd.Series,
    eth_close: pd.Series,
    lookback: int = 42,
    entry_z: float = 2.0,
    exit_z: float = 0.5,
    leg_weight: float = 0.5,
) -> pd.DataFrame:
    """Return DataFrame with columns ['BTCUSDT', 'ETHUSDT'] of weights.

    Positive weight on BTC + negative on ETH means long BTC, short ETH.
    """
    if lookback < 5:
        raise ValueError("lookback must be >= 5")
    if entry_z <= 0 or exit_z < 0:
        raise ValueError("entry_z must be > 0, exit_z >= 0")
    if not 0 < leg_weight <= 1:
        raise ValueError("leg_weight must be in (0, 1]")

    # Align.
    df = pd.DataFrame({"btc": btc_close, "eth": eth_close}).dropna()
    df["ratio"] = np.log(df["btc"] / df["eth"])

    mean = df["ratio"].rolling(lookback, min_periods=lookback).mean()
    std = df["ratio"].rolling(lookback, min_periods=lookback).std()
    df["z"] = (df["ratio"] - mean) / std.replace(0, np.nan)

    weights = pd.DataFrame(0.0, index=df.index, columns=["BTCUSDT", "ETHUSDT"])

    z = df["z"].to_numpy()
    n = len(df)
    state = 0  # -1 = short BTC/long ETH, 0 = flat, +1 = long BTC/short ETH

    for i in range(n):
        zi = z[i]
        if np.isnan(zi):
            weights.iloc[i] = [0.0, 0.0]
            continue

        if state == 0:
            if zi > entry_z:
                state = -1  # BTC expensive → short BTC, long ETH
            elif zi < -entry_z:
                state = +1  # BTC cheap → long BTC, short ETH
        elif state == +1:
            if abs(zi) < exit_z:
                state = 0
        elif state == -1:
            if abs(zi) < exit_z:
                state = 0

        if state == +1:
            weights.iloc[i] = [leg_weight, -leg_weight]  # long BTC, short ETH
        elif state == -1:
            weights.iloc[i] = [-leg_weight, leg_weight]  # short BTC, long ETH
        # else flat: [0, 0]

    # Signal was computed from close at bar i, but execution must happen at
    # bar i+1's open. The portfolio simulator already lags weights by 1 bar,
    # so we return the raw signal. But we must NOT use bars where the ratio
    # itself depends on future info. The rolling mean/std are causal, and the
    # state machine uses only z[i]. Execution lag is handled downstream.
    return weights


def pair_spread_weights(
    btc_close: pd.Series,
    eth_close: pd.Series,
    lookback: int = 42,
    entry_z: float = 2.0,
    exit_z: float = 0.5,
    leg_weight: float = 0.5,
    symbol_a: str = "BTCUSDT",
    symbol_b: str = "ETHUSDT",
) -> pd.DataFrame:
    """Return DataFrame with columns [symbol_a, symbol_b] of weights.

    Positive weight on symbol_a + negative on symbol_b means long a, short b.
    """
    if lookback < 5:
        raise ValueError("lookback must be >= 5")
    if entry_z <= 0 or exit_z < 0:
        raise ValueError("entry_z must be > 0, exit_z >= 0")
    if not 0 < leg_weight <= 1:
        raise ValueError("leg_weight must be in (0, 1]")

    df = pd.DataFrame({"a": btc_close, "b": eth_close}).dropna()
    df["ratio"] = np.log(df["a"] / df["b"])

    mean = df["ratio"].rolling(lookback, min_periods=lookback).mean()
    std = df["ratio"].rolling(lookback, min_periods=lookback).std()
    df["z"] = (df["ratio"] - mean) / std.replace(0, np.nan)

    weights = pd.DataFrame(0.0, index=df.index, columns=[symbol_a, symbol_b])

    z = df["z"].to_numpy()
    n = len(df)
    state = 0

    for i in range(n):
        zi = z[i]
        if np.isnan(zi):
            weights.iloc[i] = [0.0, 0.0]
            continue

        if state == 0:
            if zi > entry_z:
                state = -1
            elif zi < -entry_z:
                state = +1
        elif state == +1:
            if abs(zi) < exit_z:
                state = 0
        elif state == -1:
            if abs(zi) < exit_z:
                state = 0

        if state == +1:
            weights.iloc[i] = [leg_weight, -leg_weight]
        elif state == -1:
            weights.iloc[i] = [-leg_weight, leg_weight]

    return weights


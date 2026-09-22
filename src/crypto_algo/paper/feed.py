"""Fetch recent klines from Binance USDT-M futures public REST."""

from __future__ import annotations

from datetime import datetime, timezone

import httpx
import pandas as pd

from crypto_algo.data.futures_fetcher import KLINE_COLUMNS, NUMERIC_COLUMNS

FUTURES_BASE = "https://fapi.binance.com"


def fetch_recent_klines(symbol: str, interval: str, limit: int = 500) -> pd.DataFrame:
    """Fetch the most recent `limit` klines. Returns DataFrame in our standard schema."""
    url = f"{FUTURES_BASE}/fapi/v1/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    r = httpx.get(url, params=params, timeout=15.0)
    r.raise_for_status()
    data = r.json()

    df = pd.DataFrame(data, columns=KLINE_COLUMNS).drop(columns=["ignore"])
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
    for c in NUMERIC_COLUMNS:
        df[c] = df[c].astype(float)
    df["num_trades"] = df["num_trades"].astype(int)
    df["symbol"] = symbol
    df["interval"] = interval
    df["source"] = "binance_futures"
    df["fetched_at"] = datetime.now(timezone.utc)
    return df.sort_values("open_time").reset_index(drop=True)

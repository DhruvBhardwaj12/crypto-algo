"""Deribit DVOL fetcher.

DVOL is Deribit's 30-day forward-looking implied volatility index for
BTC and ETH. It's the crypto equivalent of VIX.

Public API, no key needed.

Endpoint: /api/v2/public/get_volatility_index_data
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import httpx
import pandas as pd
from loguru import logger

DERIBIT_BASE = "https://www.deribit.com/api/v2"


def _to_ms(date_str: str) -> int:
    dt = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def fetch_dvol(
    currency: str,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """Fetch DVOL daily closes. Returns DataFrame with timestamp + close.

    Currency: 'BTC' or 'ETH'.
    DVOL is quoted as an annualized percent (e.g., 55.3 means 55.3% IV).
    """
    url = f"{DERIBIT_BASE}/public/get_volatility_index_data"
    params = {
        "currency": currency,
        "start_timestamp": _to_ms(start_date),
        "end_timestamp": _to_ms(end_date),
        "resolution": "1D",
    }

    with httpx.Client(timeout=30.0) as client:
        r = client.get(url, params=params)
        r.raise_for_status()
        payload = r.json()

    if "result" not in payload or "data" not in payload["result"]:
        raise ValueError(f"Unexpected DVOL response: {str(payload)[:300]}")

    rows = payload["result"]["data"]
    # rows = [[timestamp_ms, open, high, low, close], ...]
    df = pd.DataFrame(rows, columns=["ts_ms", "open", "high", "low", "close"])
    df["timestamp"] = pd.to_datetime(df["ts_ms"], unit="ms", utc=True)
    df = df[["timestamp", "open", "high", "low", "close"]].sort_values("timestamp").reset_index(drop=True)
    df["currency"] = currency
    df["source"] = "deribit"

    logger.success("Fetched {} DVOL rows for {} ({} to {})",
                   len(df), currency, df["timestamp"].iloc[0], df["timestamp"].iloc[-1])
    return df


def save_dvol(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False, compression="snappy")
    logger.success("Saved DVOL to {}", path)


    
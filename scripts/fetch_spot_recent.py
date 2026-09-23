"""Fetch spot OHLCV for BTC/ETH. Needed for the funding carry long leg.

Run:
    uv run python scripts/fetch_spot_recent.py
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path

import httpx
import pandas as pd
from loguru import logger

from crypto_algo.data.futures_fetcher import KLINE_COLUMNS, NUMERIC_COLUMNS, save_parquet

SPOT_BASE = "https://data-api.binance.vision"
SYMBOLS = ["BTCUSDT", "ETHUSDT"]
INTERVAL = "4h"
START = "2024-01-01"
END = "2026-09-23"
OUT_DIR = Path("data/raw/binance/spot")


def _to_ms(date_str: str) -> int:
    return int(datetime.strptime(date_str, "%Y-%m-%d")
               .replace(tzinfo=timezone.utc).timestamp() * 1000)


def fetch_spot(symbol: str, start_ms: int, end_ms: int, limit: int = 1000) -> pd.DataFrame:
    url = f"{SPOT_BASE}/api/v3/klines"
    all_rows = []
    cursor = start_ms
    with httpx.Client() as client:
        while cursor < end_ms:
            params = {"symbol": symbol, "interval": INTERVAL,
                      "startTime": cursor, "endTime": end_ms, "limit": limit}
            r = client.get(url, params=params, timeout=30.0)
            r.raise_for_status()
            page = r.json()
            if not page:
                break
            all_rows.extend(page)
            last = int(page[-1][0])
            if last <= cursor:
                break
            cursor = last + 1
            time.sleep(0.3)
            if len(page) < limit:
                break

    df = pd.DataFrame(all_rows, columns=KLINE_COLUMNS).drop(columns=["ignore"])
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
    for c in NUMERIC_COLUMNS:
        df[c] = df[c].astype(float)
    df["num_trades"] = df["num_trades"].astype(int)
    df = df.sort_values("open_time").drop_duplicates("open_time").reset_index(drop=True)
    df["symbol"] = symbol
    df["interval"] = INTERVAL
    df["source"] = "binance_spot"
    df["fetched_at"] = datetime.now(timezone.utc)
    return df


def main() -> None:
    start_ms = _to_ms(START)
    end_ms = _to_ms(END)
    for symbol in SYMBOLS:
        logger.info("Fetching spot {} from {} to {}", symbol, START, END)
        df = fetch_spot(symbol, start_ms, end_ms)
        out = OUT_DIR / symbol / INTERVAL / f"{START}_{END}.parquet"
        save_parquet(df, out)


if __name__ == "__main__":
    main()

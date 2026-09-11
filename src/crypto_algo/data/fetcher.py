"""Binance public market data fetcher.

Fetches historical OHLCV klines from Binance's public REST API.
No API key required — this only touches public market data endpoints.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
import pandas as pd
from loguru import logger

KLINE_COLUMNS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
    "num_trades",
    "taker_buy_base_volume",
    "taker_buy_quote_volume",
    "ignore",
]

NUMERIC_COLUMNS = [
    "open",
    "high",
    "low",
    "close",
    "volume",
    "quote_volume",
    "taker_buy_base_volume",
    "taker_buy_quote_volume",
]


def _to_millis(date_str: str) -> int:
    """Convert 'YYYY-MM-DD' to milliseconds since Unix epoch (UTC)."""
    dt = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def _fetch_klines_page(
    client: httpx.Client,
    base_url: str,
    symbol: str,
    interval: str,
    start_ms: int,
    end_ms: int,
    limit: int,
) -> list[list[Any]]:
    """Fetch a single page of klines. Returns the raw list-of-lists."""
    url = f"{base_url}/api/v3/klines"
    params = {
        "symbol": symbol,
        "interval": interval,
        "startTime": start_ms,
        "endTime": end_ms,
        "limit": limit,
    }

    response = client.get(url, params=params, timeout=30.0)
    response.raise_for_status()
    data = response.json()

    if not isinstance(data, list):
        raise ValueError(
            f"Unexpected response type from Binance: {type(data)}. "
            f"Body: {str(data)[:200]}"
        )

    return data


def fetch_klines(
    base_url: str,
    symbol: str,
    interval: str,
    start_date: str,
    end_date: str,
    request_limit: int = 1000,
    request_delay_seconds: float = 0.3,
) -> pd.DataFrame:
    """Fetch all klines for a symbol between two dates (inclusive of start, exclusive of end)."""
    start_ms = _to_millis(start_date)
    end_ms = _to_millis(end_date)

    all_rows: list[list[Any]] = []
    cursor_ms = start_ms

    logger.info(
        "Fetching {} {} from {} to {}",
        symbol,
        interval,
        start_date,
        end_date,
    )

    with httpx.Client() as client:
        while cursor_ms < end_ms:
            page = _fetch_klines_page(
                client=client,
                base_url=base_url,
                symbol=symbol,
                interval=interval,
                start_ms=cursor_ms,
                end_ms=end_ms,
                limit=request_limit,
            )

            if not page:
                logger.info("No more data returned. Stopping.")
                break

            all_rows.extend(page)

            last_open_time = int(page[-1][0])
            if last_open_time <= cursor_ms:
                logger.warning("Cursor did not advance. Stopping to avoid infinite loop.")
                break

            cursor_ms = last_open_time + 1

            logger.info(
                "  fetched page of {} candles, total so far: {}",
                len(page),
                len(all_rows),
            )

            time.sleep(request_delay_seconds)

            if len(page) < request_limit:
                break

    if not all_rows:
        raise ValueError(f"No data returned for {symbol} between {start_date} and {end_date}.")

    df = pd.DataFrame(all_rows, columns=KLINE_COLUMNS)
    df = df.drop(columns=["ignore"])

    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)

    for col in NUMERIC_COLUMNS:
        df[col] = df[col].astype(float)

    df["num_trades"] = df["num_trades"].astype(int)

    df = df.sort_values("open_time").drop_duplicates(subset=["open_time"]).reset_index(drop=True)

    df["symbol"] = symbol
    df["interval"] = interval
    df["source"] = "binance"
    df["fetched_at"] = datetime.now(timezone.utc)

    logger.success("Fetched {} rows for {}", len(df), symbol)
    return df


def save_raw(df: pd.DataFrame, output_path: Path) -> None:
    """Save a DataFrame to Parquet, creating parent directories if needed."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False, compression="snappy")
    logger.success("Saved {} rows to {}", len(df), output_path)
    
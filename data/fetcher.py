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

# Binance returns klines as arrays in this exact order.
# We map each position to a named column.
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

# Columns that must be numeric (Binance sends them as strings).
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
    """Fetch all klines for a symbol between two dates (inclusive of start, exclusive of end).

    Handles pagination automatically: fetches in batches of `request_limit`
    until we reach `end_date` or Binance stops returning data.
    """
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

            # Advance cursor to just after the last candle's open time.
            last_open_time = int(page[-1][0])
            if last_open_time <= cursor_ms:
                # Safety: if the API returns the same timestamp, stop to avoid infinite loop.
                logger.warning("Cursor did not advance. Stopping to avoid infinite loop.")
                break

            cursor_ms = last_open_time + 1  # +1 ms to avoid re-fetching the same candle

            logger.info(
                "  fetched page of {} candles, total so far: {}",
                len(page),
                len(all_rows),
            )

            # Be polite. Respect the API.
            time.sleep(request_delay_seconds)

            # If the last page returned fewer than the limit, we're done.
            if len(page) < request_limit:
                break

    if not all_rows:
        raise ValueError(f"No data returned for {symbol} between {start_date} and {end_date}.")

    df = pd.DataFrame(all_rows, columns=KLINE_COLUMNS)

    # Drop the useless 'ignore' column.
    df = df.drop(columns=["ignore"])

    # Convert timestamps to UTC datetimes.
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)

    # Convert price/volume strings to floats.
    for col in NUMERIC_COLUMNS:
        df[col] = df[col].astype(float)

    df["num_trades"] = df["num_trades"].astype(int)

    # Sort and deduplicate by open_time.
    df = df.sort_values("open_time").drop_duplicates(subset=["open_time"]).reset_index(drop=True)

    # Add metadata columns.
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
    
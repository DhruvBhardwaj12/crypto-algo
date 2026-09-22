"""Binance USDT-M perpetual futures public data fetcher.

Fetches from the Binance futures API (fapi.binance.com):
  - Perpetual futures OHLCV klines
  - Funding rate history (paid every 8 hours)

No API key required. These are public market data endpoints.
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
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_volume", "num_trades",
    "taker_buy_base_volume", "taker_buy_quote_volume", "ignore",
]

NUMERIC_COLUMNS = [
    "open", "high", "low", "close", "volume", "quote_volume",
    "taker_buy_base_volume", "taker_buy_quote_volume",
]

FUNDING_COLUMNS = ["funding_time", "funding_rate", "symbol"]


def _to_millis(date_str: str) -> int:
    dt = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def _fetch_klines_page(
    client: httpx.Client, base_url: str, symbol: str, interval: str,
    start_ms: int, end_ms: int, limit: int,
) -> list[list[Any]]:
    url = f"{base_url}/fapi/v1/klines"
    params = {
        "symbol": symbol, "interval": interval,
        "startTime": start_ms, "endTime": end_ms, "limit": limit,
    }
    response = client.get(url, params=params, timeout=30.0)
    response.raise_for_status()
    data = response.json()
    if not isinstance(data, list):
        raise ValueError(f"Unexpected futures kline response: {str(data)[:200]}")
    return data


def fetch_futures_klines(
    base_url: str, symbol: str, interval: str,
    start_date: str, end_date: str,
    request_limit: int = 1000, request_delay_seconds: float = 0.3,
) -> pd.DataFrame:
    """Fetch perpetual futures OHLCV. Same pagination pattern as spot."""
    start_ms = _to_millis(start_date)
    end_ms = _to_millis(end_date)

    all_rows: list[list[Any]] = []
    cursor_ms = start_ms

    logger.info("Fetching futures {} {} from {} to {}", symbol, interval, start_date, end_date)

    with httpx.Client() as client:
        while cursor_ms < end_ms:
            page = _fetch_klines_page(
                client, base_url, symbol, interval, cursor_ms, end_ms, request_limit
            )
            if not page:
                break
            all_rows.extend(page)
            last = int(page[-1][0])
            if last <= cursor_ms:
                break
            cursor_ms = last + 1
            logger.info("  page {} candles, total {}", len(page), len(all_rows))
            time.sleep(request_delay_seconds)
            if len(page) < request_limit:
                break

    if not all_rows:
        raise ValueError(f"No futures data for {symbol} {start_date}..{end_date}")

    df = pd.DataFrame(all_rows, columns=KLINE_COLUMNS).drop(columns=["ignore"])
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
    for c in NUMERIC_COLUMNS:
        df[c] = df[c].astype(float)
    df["num_trades"] = df["num_trades"].astype(int)
    df = df.sort_values("open_time").drop_duplicates(subset=["open_time"]).reset_index(drop=True)
    df["symbol"] = symbol
    df["interval"] = interval
    df["source"] = "binance_futures"
    df["fetched_at"] = datetime.now(timezone.utc)

    logger.success("Fetched {} futures rows for {}", len(df), symbol)
    return df


def _fetch_funding_page(
    client: httpx.Client, base_url: str, symbol: str,
    start_ms: int, end_ms: int, limit: int,
) -> list[dict]:
    url = f"{base_url}/fapi/v1/fundingRate"
    params = {
        "symbol": symbol, "startTime": start_ms, "endTime": end_ms, "limit": limit,
    }
    response = client.get(url, params=params, timeout=30.0)
    response.raise_for_status()
    data = response.json()
    if not isinstance(data, list):
        raise ValueError(f"Unexpected funding response: {str(data)[:200]}")
    return data


def fetch_funding_rates(
    base_url: str, symbol: str,
    start_date: str, end_date: str,
    request_limit: int = 1000, request_delay_seconds: float = 0.3,
) -> pd.DataFrame:
    """Fetch funding rate history. Binance returns <= 1000 records per request.

    Funding occurs every 8 hours, so ~1095 records/year. Pagination is
    essential for multi-year history.
    """
    start_ms = _to_millis(start_date)
    end_ms = _to_millis(end_date)
    cursor_ms = start_ms

    all_rows: list[dict] = []
    logger.info("Fetching funding rates {} from {} to {}", symbol, start_date, end_date)

    with httpx.Client() as client:
        while cursor_ms < end_ms:
            page = _fetch_funding_page(
                client, base_url, symbol, cursor_ms, end_ms, request_limit
            )
            if not page:
                break
            all_rows.extend(page)
            last = int(page[-1]["fundingTime"])
            if last <= cursor_ms:
                break
            cursor_ms = last + 1
            logger.info("  funding page {} records, total {}", len(page), len(all_rows))
            time.sleep(request_delay_seconds)
            if len(page) < request_limit:
                break

    if not all_rows:
        raise ValueError(f"No funding data for {symbol}")

    df = pd.DataFrame(all_rows)
    df = df.rename(columns={"fundingTime": "funding_time", "fundingRate": "funding_rate"})
    df["funding_time"] = pd.to_datetime(df["funding_time"], unit="ms", utc=True)
    df["funding_rate"] = df["funding_rate"].astype(float)
    df["symbol"] = symbol
    df = df[["funding_time", "funding_rate", "symbol"]]
    df = df.sort_values("funding_time").drop_duplicates(subset=["funding_time"]).reset_index(drop=True)
    df["fetched_at"] = datetime.now(timezone.utc)

    logger.success("Fetched {} funding records for {}", len(df), symbol)
    return df


def save_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False, compression="snappy")
    logger.success("Saved {} rows to {}", len(df), path)
    
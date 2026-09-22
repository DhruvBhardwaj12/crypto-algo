"""Fetch Binance USDT-M futures klines and funding rate history.

Fetches all intervals listed in `futures_intervals_to_fetch`.

Run:
    uv run python scripts/fetch_futures_data.py
"""

from __future__ import annotations

from pathlib import Path

import yaml
from loguru import logger

from crypto_algo.data.futures_fetcher import (
    fetch_funding_rates,
    fetch_futures_klines,
    save_parquet,
)


def main() -> None:
    with open("config/settings.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)["data"]

    base = cfg["futures_base_url"]
    raw = Path(cfg["raw_data_dir"])
    intervals = cfg["futures_intervals_to_fetch"]
    start = cfg["start_date"]
    end = cfg["end_date"]

    for symbol in cfg["symbols"]:
        for interval in intervals:
            klines = fetch_futures_klines(
                base_url=base,
                symbol=symbol,
                interval=interval,
                start_date=start,
                end_date=end,
                request_limit=cfg["request_limit"],
                request_delay_seconds=cfg["request_delay_seconds"],
            )
            save_parquet(
                klines,
                raw / "binance" / "futures" / symbol / interval / f"{start}_{end}.parquet",
            )

        funding = fetch_funding_rates(
            base_url=base,
            symbol=symbol,
            start_date=start,
            end_date=end,
            request_limit=cfg["request_limit"],
            request_delay_seconds=cfg["request_delay_seconds"],
        )
        save_parquet(
            funding,
            raw / "binance" / "funding" / symbol / f"{start}_{end}.parquet",
        )

    logger.success("All futures data fetched.")


if __name__ == "__main__":
    main()
    
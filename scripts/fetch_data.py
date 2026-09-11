"""Fetch historical OHLCV data from Binance and save as Parquet."""

from __future__ import annotations

from pathlib import Path

import yaml
from loguru import logger

from crypto_algo.data.fetcher import fetch_klines, save_raw


def main() -> None:
    config_path = Path("config/settings.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    data_cfg = config["data"]
    raw_dir = Path(data_cfg["raw_data_dir"])

    for symbol in data_cfg["symbols"]:
        df = fetch_klines(
            base_url=data_cfg["base_url"],
            symbol=symbol,
            interval=data_cfg["interval"],
            start_date=data_cfg["start_date"],
            end_date=data_cfg["end_date"],
            request_limit=data_cfg["request_limit"],
            request_delay_seconds=data_cfg["request_delay_seconds"],
        )

        output_path = (
            raw_dir
            / data_cfg["exchange"]
            / symbol
            / data_cfg["interval"]
            / f"{data_cfg['start_date']}_{data_cfg['end_date']}.parquet"
        )
        save_raw(df, output_path)

    logger.success("All symbols fetched.")


if __name__ == "__main__":
    main()
"""Fetch DVOL for BTC and ETH from Deribit.

Run:
    uv run python scripts/fetch_dvol.py
"""

from __future__ import annotations

from pathlib import Path

from crypto_algo.data.deribit_fetcher import fetch_dvol, save_dvol

START = "2021-01-01"   # DVOL launched in early 2021
END = "2026-09-23"
OUT_DIR = Path("data/raw/deribit/dvol")


def main() -> None:
    for currency in ["BTC", "ETH"]:
        df = fetch_dvol(currency, START, END)
        out = OUT_DIR / f"{currency}_{START}_{END}.parquet"
        save_dvol(df, out)


if __name__ == "__main__":
    main()
    
"""Validate all raw parquet files and print a report per symbol.

Run:
    uv run python scripts/validate_data.py
"""

from __future__ import annotations

from pathlib import Path

from crypto_algo.data.loaders import load_parquet
from crypto_algo.data.validation import validate_ohlcv


def main() -> None:
    raw_dir = Path("data/raw/binance")
    files = sorted(raw_dir.rglob("*.parquet"))
    if not files:
        print(f"No parquet files under {raw_dir}. Run scripts/fetch_data.py first.")
        return

    any_dirty = False
    for path in files:
        df = load_parquet(path)
        report = validate_ohlcv(df, interval=str(df["interval"].iloc[0]))
        print(report.summary())
        print()
        if not report.is_clean:
            any_dirty = True
            if report.missing_intervals:
                print(f"  First 5 gaps in {report.symbol}:")
                for ts in report.missing_intervals[:5]:
                    print(f"    {ts}")
                print()

    if any_dirty:
        print("Some datasets have issues. Record them in docs/decisions.md.")
    else:
        print("All datasets are clean.")


if __name__ == "__main__":
    main()
    
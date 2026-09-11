"""Sanity check on fetched data.

Confirms row counts, gaps, duplicates, and value ranges look correct.
Run: uv run python scripts/inspect_data.py
"""

from pathlib import Path

import pandas as pd


def inspect(path: Path) -> None:
    print(f"\n=== {path.name} ===")
    df = pd.read_parquet(path)

    print(f"Rows: {len(df)}")
    print(f"Columns ({len(df.columns)}): {list(df.columns)}")
    print(f"Date range: {df['open_time'].min()} → {df['open_time'].max()}")
    print(f"All timestamps unique: {df['open_time'].is_unique}")
    print(f"Timestamps sorted ascending: {df['open_time'].is_monotonic_increasing}")
    print(f"\nMissing values per column:\n{df.isna().sum().to_string()}")

    # Gap analysis: count hours between min and max, compare to actual rows.
    expected = pd.date_range(df["open_time"].min(), df["open_time"].max(), freq="h")
    missing = expected.difference(df["open_time"])
    print(f"\nExpected hourly candles: {len(expected)}")
    print(f"Actual rows:             {len(df)}")
    print(f"Missing hourly candles:  {len(missing)}")
    if len(missing) > 0:
        print(f"First 5 missing: {list(missing[:5])}")

    # Price sanity: close must be positive and not absurd.
    print(f"\nClose price range: {df['close'].min():.2f} → {df['close'].max():.2f}")
    print(f"Any close <= 0: {(df['close'] <= 0).any()}")
    print(f"Any volume < 0: {(df['volume'] < 0).any()}")

    # OHLC sanity: high must be >= low, and high >= max(open, close).
    bad_high = (df["high"] < df[["open", "close", "low"]].max(axis=1)).sum()
    bad_low = (df["low"] > df[["open", "close", "high"]].min(axis=1)).sum()
    print(f"Rows where high < something else: {bad_high}")
    print(f"Rows where low > something else:  {bad_low}")


def main() -> None:
    raw_dir = Path("data/raw/binance")
    files = sorted(raw_dir.rglob("*.parquet"))
    if not files:
        print(f"No parquet files found under {raw_dir}")
        return
    for f in files:
        inspect(f)


if __name__ == "__main__":
    main()
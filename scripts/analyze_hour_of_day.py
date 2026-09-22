"""Exploratory analysis: does hour-of-day matter in BTC/ETH futures?

For each hour of the day (0-23 UTC), we compute the distribution of 1H
returns during that hour across 5 years of data.

Outputs:
  - per-hour mean, std, t-stat, n
  - year-by-year breakdown for stability check
  - split-half test: first half vs second half

No strategy is designed until we see the results.

Run:
    uv run python scripts/analyze_hour_of_day.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from crypto_algo.data.loaders import load_parquet

START = "2020-01-01"
END = "2025-01-01"
SYMBOLS = ["BTCUSDT", "ETHUSDT"]


def _load(symbol: str) -> pd.DataFrame:
    return load_parquet(
        Path(f"data/raw/binance/futures/{symbol}/1h/{START}_{END}.parquet")
    )


def _hourly_stats(returns: pd.Series) -> pd.DataFrame:
    """For each hour of day, compute mean/std/t-stat/n of log returns."""
    df = pd.DataFrame({"ret": returns})
    df["hour"] = df.index.hour
    grouped = df.groupby("hour")["ret"]

    out = pd.DataFrame({
        "n": grouped.count(),
        "mean_bps": grouped.mean() * 10_000.0,
        "std_bps": grouped.std() * 10_000.0,
    })
    out["stderr_bps"] = out["std_bps"] / np.sqrt(out["n"])
    out["t_stat"] = out["mean_bps"] / out["stderr_bps"]
    out["annualized_pct"] = out["mean_bps"] * 24 * 365 / 100.0
    return out.reset_index()


def _fmt_row(row: pd.Series) -> str:
    return (
        f"  {int(row['hour']):02d}h  "
        f"n={int(row['n']):5d}  "
        f"mean={row['mean_bps']:+7.3f} bps  "
        f"std={row['std_bps']:6.1f}  "
        f"t={row['t_stat']:+6.2f}  "
        f"ann_if_always={row['annualized_pct']:+8.2f}%"
    )


def main() -> None:
    for symbol in SYMBOLS:
        df = _load(symbol).copy()
        df = df.sort_values("open_time").set_index("open_time")
        # Log return of each 1H bar (open to close, or close to close? Use close-to-close)
        df["ret"] = np.log(df["close"] / df["close"].shift(1))
        df = df.dropna(subset=["ret"])

        print(f"\n{'=' * 90}")
        print(f"  {symbol}  —  hour-of-day analysis  —  {START} to {END}")
        print(f"{'=' * 90}")

        # Full sample
        stats = _hourly_stats(df["ret"])
        print("\nFull-sample hour-of-day stats (UTC):")
        for _, row in stats.iterrows():
            print(_fmt_row(row))

        # Bonferroni-corrected significance threshold: |t| > 3.1 for 24 tests at 5%.
        sig_hours = stats[stats["t_stat"].abs() > 3.1]
        if len(sig_hours) > 0:
            print(f"\n  Hours passing Bonferroni threshold (|t| > 3.1):")
            for _, row in sig_hours.iterrows():
                print(f"    hour {int(row['hour']):02d}: t={row['t_stat']:+.2f}, mean={row['mean_bps']:+.3f} bps")
        else:
            print("\n  No hours pass Bonferroni threshold. Likely noise.")

        # Split-half test: does the pattern hold on each half?
        mid = df.index[len(df) // 2]
        first_half = df[df.index < mid]
        second_half = df[df.index >= mid]

        stats_1h = _hourly_stats(first_half["ret"])
        stats_2h = _hourly_stats(second_half["ret"])

        merged = stats_1h[["hour", "mean_bps", "t_stat"]].merge(
            stats_2h[["hour", "mean_bps", "t_stat"]],
            on="hour",
            suffixes=("_1st", "_2nd"),
        )
        merged["signs_agree"] = np.sign(merged["mean_bps_1st"]) == np.sign(merged["mean_bps_2nd"])
        n_agree = int(merged["signs_agree"].sum())

        print(f"\nSplit-half consistency:")
        print(f"  split at: {mid}")
        print(f"  hours where both halves have same-sign mean: {n_agree}/24")
        print(f"  (expect ~12 by chance; higher indicates a persistent pattern)")
        print()
        print(f"  {'hour':>4s}  {'1st half':>12s}  {'2nd half':>12s}  {'agree?':>7s}")
        for _, row in merged.iterrows():
            print(
                f"  {int(row['hour']):>4d}  "
                f"{row['mean_bps_1st']:>+11.3f}  "
                f"{row['mean_bps_2nd']:>+11.3f}  "
                f"{'yes' if row['signs_agree'] else 'no':>7s}"
            )

        # Year-by-year breakdown
        print(f"\nYear-by-year mean bps by hour:")
        df_reset = df.copy()
        df_reset["year"] = df_reset.index.year
        df_reset["hour"] = df_reset.index.hour
        pivot = df_reset.pivot_table(
            index="hour", columns="year", values="ret", aggfunc="mean"
        ) * 10_000.0
        with pd.option_context("display.width", 200, "display.max_columns", 20):
            print(pivot.round(3).to_string())


if __name__ == "__main__":
    main()
    
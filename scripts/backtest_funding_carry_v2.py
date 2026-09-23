"""Backtest funding carry v2 on BTC/ETH, both windows.

Run:
    uv run python scripts/backtest_funding_carry_v2.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from loguru import logger

from crypto_algo.backtesting.carry_simulator_v2 import run_carry_v2
from crypto_algo.backtesting.futures_costs import FuturesCostModel
from crypto_algo.data.loaders import load_parquet
from crypto_algo.strategies.funding_carry_v2 import funding_carry_signal_v2

INITIAL_EQUITY = 10_000.0
COST = FuturesCostModel()


def _load_pair(symbol: str, start: str, end: str):
    # Spot
    spot_path = Path(f"data/raw/binance/spot/{symbol}/4h/{start}_{end}.parquet")
    if not spot_path.exists():
        fallback = Path(f"data/raw/binance/spot/{symbol}/4h/2024-01-01_2026-09-23.parquet")
        if fallback.exists():
            spot_path = fallback
        else:
            return None
    spot = load_parquet(spot_path)

    # Perp 4h
    perp_path = Path(f"data/raw/binance/futures/{symbol}/4h/{start}_{end}.parquet")
    if not perp_path.exists():
        return None
    perp = load_parquet(perp_path)

    # Funding — different schema than OHLCV, so bypass load_parquet.
    fund_path = Path(f"data/raw/binance/funding/{symbol}/{start}_{end}.parquet")
    if not fund_path.exists():
        return None
    funding = pd.read_parquet(fund_path)

    # Sanity: verify all three frames cover the same date range.
    spot_min, spot_max = spot["open_time"].min(), spot["open_time"].max()
    perp_min, perp_max = perp["open_time"].min(), perp["open_time"].max()
    fund_min, fund_max = funding["funding_time"].min(), funding["funding_time"].max()

    print(f"    [sanity] {symbol} spot:    {spot_min} to {spot_max}")
    print(f"    [sanity] {symbol} perp:    {perp_min} to {perp_max}")
    print(f"    [sanity] {symbol} funding: {fund_min} to {fund_max}")

    latest_start = max(spot_min, perp_min, fund_min)
    earliest_end = min(spot_max, perp_max, fund_max)
    overlap_days = (earliest_end - latest_start).days
    if overlap_days < 90:
        print(f"    [sanity] INSUFFICIENT OVERLAP: {overlap_days} days. Skipping.")
        return None

    return spot, perp, funding




def _resample_funding_grid(
    funding: pd.DataFrame, spot: pd.DataFrame, perp: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Align spot and perp to funding timestamps."""
    f = funding.copy().sort_values("funding_time").reset_index(drop=True)
    s = spot.copy().sort_values("open_time").reset_index(drop=True)
    p = perp.copy().sort_values("open_time").reset_index(drop=True)

    # For each funding timestamp, find the most recent closed bar.
    # Reuse merge_asof on close_time.
    f_sorted = f.sort_values("funding_time").reset_index(drop=True)
    merged_s = pd.merge_asof(
        f_sorted[["funding_time"]],
        s[["close_time", "close"]].rename(columns={"close": "spot_close"}),
        left_on="funding_time", right_on="close_time", direction="backward",
    )
    merged_p = pd.merge_asof(
        f_sorted[["funding_time"]],
        p[["close_time", "close"]].rename(columns={"close": "perp_close"}),
        left_on="funding_time", right_on="close_time", direction="backward",
    )
    # Rebuild DataFrames matching run_carry_v2's expected shape.
    spot_df = merged_s[["funding_time", "spot_close"]].dropna().rename(columns={"spot_close": "close"})
    perp_df = merged_p[["funding_time", "perp_close"]].dropna().rename(columns={"perp_close": "close"})
    # Funding: keep only rows with valid spot and perp.
    funding_df = f_sorted[["funding_time", "funding_rate"]].iloc[:len(spot_df)].reset_index(drop=True)
    spot_df = spot_df.reset_index(drop=True)
    perp_df = perp_df.reset_index(drop=True)

    n = min(len(spot_df), len(perp_df), len(funding_df))
    print(f"    [debug] aligned lengths: spot={len(spot_df)} perp={len(perp_df)} funding={len(funding_df)}")
    print(f"    [debug] first 3 rows:")
    for j in range(min(3, n)):
        print(f"      funding_time={funding_df['funding_time'].iloc[j]}  "
              f"spot={spot_df['close'].iloc[j]:.2f}  perp={perp_df['close'].iloc[j]:.2f}  "
              f"basis={(perp_df['close'].iloc[j] - spot_df['close'].iloc[j]) / spot_df['close'].iloc[j]:.6f}")
    return (
        spot_df.iloc[:n].reset_index(drop=True),
        perp_df.iloc[:n].reset_index(drop=True),
        funding_df.iloc[:n].reset_index(drop=True),
    )


def _fmt(v: float) -> str:
    if v != v:
        return "n/a"
    return f"{v:+.3f}" if abs(v) < 10 else f"{v:+,.2f}"


def main() -> None:
    windows = [
    ("HISTORICAL 2020-2025", "2020-01-01", "2025-01-01"),
    ("RECENT 2025-2026",     "2025-01-01", "2026-09-23"),
]
    for symbol in ["BTCUSDT", "ETHUSDT"]:
        print(f"\n{'=' * 90}")
        print(f"  Funding Carry v2 — {symbol}")
        print(f"{'=' * 90}")

        for label, start, end in windows:
            data = _load_pair(symbol, start, end)
            if data is None:
                print(f"\n  [{label}] skipped — missing data")
                continue
            spot, perp, funding = data
            aligned_spot, aligned_perp, aligned_fund = _resample_funding_grid(funding, spot, perp)
            if len(aligned_fund) < 100:
                print(f"\n  [{label}] insufficient aligned bars: {len(aligned_fund)}")
                continue

            sig = funding_carry_signal_v2(aligned_fund["funding_rate"])
            sig.index = aligned_fund.index

            result = run_carry_v2(
                spot=aligned_spot,
                perp=aligned_perp,
                funding=aligned_fund,
                signal=sig,
                cost_model=COST,
                initial_equity=INITIAL_EQUITY,
            )

            print(f"\n  [{label}]  {aligned_fund['funding_time'].iloc[0].date()} to {aligned_fund['funding_time'].iloc[-1].date()}")
            print(f"    Bars:                {len(aligned_fund)}")
            print(f"    Entries:             {result.n_entries}")
            print(f"    Funding income:      ${result.total_funding:>+10,.2f}")
            print(f"    Basis P&L:           ${result.total_basis:>+10,.2f}")
            print(f"    Costs paid:          ${result.total_costs:>+10,.2f}")
            print(f"    Final equity:        ${result.final_equity:>+10,.2f}")
            print(f"    Metrics:")
            for k, v in result.metrics.items():
                if isinstance(v, float):
                    print(f"      {k:18s} {_fmt(v)}")
                else:
                    print(f"      {k:18s} {v}")


if __name__ == "__main__":
    main()

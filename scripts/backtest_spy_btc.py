"""Lab-005: S&P 500 cross-asset momentum on BTC.

Hypothesis: During US market hours (13:30-20:00 UTC), BTC returns are
positively predicted by recent S&P 500 returns. Cross-asset information
flow via shared institutional participants and risk-on/risk-off sentiment.

Two-part analysis:
  1. Correlation analysis: does lagged SPY return predict next BTC return?
  2. If yes, run a simple strategy: long BTC when SPY trailing return > 0.

Data:
  SPY 1H bars from Yahoo Finance (last 730 days).
  BTC 1H bars from our cached futures data.

Run:
    uv run python scripts/backtest_spy_btc.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from loguru import logger

from crypto_algo.backtesting.simulator import run_backtest
from crypto_algo.data.loaders import load_parquet

BTC_START = "2024-10-01"
BTC_END = "2026-09-23"
INITIAL_EQUITY = 120.0
FUTURES_TAKER_BPS = 5.0
PERIODS_PER_YEAR = 24 * 365


class _Cost:
    def __init__(self, bps: float):
        self.cost_bps = bps

    @property
    def buy_cost_bps(self) -> float:
        return self.cost_bps

    @property
    def sell_cost_bps(self) -> float:
        return self.cost_bps

    @property
    def round_trip_bps(self) -> float:
        return 2 * self.cost_bps

    def cost_fraction(self, side: str) -> float:
        return self.cost_bps / 10_000.0


COST = _Cost(FUTURES_TAKER_BPS)


def fetch_spy() -> pd.DataFrame:
    """Fetch SPY 1H bars. Returns DataFrame with UTC open_time and close_time."""
    logger.info("Fetching SPY 1H bars from Yahoo Finance")
    ticker = yf.Ticker("SPY")
    df = ticker.history(period="730d", interval="1h", auto_adjust=False)

    if df.empty:
        raise ValueError("Yahoo returned no SPY data.")

    df = df.reset_index()
    time_col = "Datetime" if "Datetime" in df.columns else df.columns[0]
    df = df.rename(columns={time_col: "open_time"})

    if df["open_time"].dt.tz is None:
        df["open_time"] = df["open_time"].dt.tz_localize("UTC")
    else:
        df["open_time"] = df["open_time"].dt.tz_convert("UTC")

    df["close_time"] = df["open_time"] + pd.Timedelta(hours=1) - pd.Timedelta(microseconds=1)
    df = df.rename(columns={"Close": "spy_close", "Open": "spy_open"})
    df = df[["open_time", "close_time", "spy_open", "spy_close"]].sort_values("close_time").reset_index(drop=True)
    df["spy_ret_1h"] = np.log(df["spy_close"] / df["spy_close"].shift(1))
    df["spy_ret_3h"] = np.log(df["spy_close"] / df["spy_close"].shift(3))

    logger.success("Fetched {} SPY bars: {} to {}", len(df), df["open_time"].iloc[0], df["open_time"].iloc[-1])
    return df


def load_btc() -> pd.DataFrame:
    path = Path(f"data/raw/binance/futures/BTCUSDT/1h/{BTC_START}_{BTC_END}.parquet")
    if path.exists():
        return load_parquet(path)

    # Fall back to any 1h file we have.
    fallback = Path("data/raw/binance/futures/BTCUSDT/1h/2020-01-01_2025-01-01.parquet")
    if fallback.exists():
        logger.warning("Using fallback BTC file: {}", fallback)
        return load_parquet(fallback)
    raise FileNotFoundError("No BTC 1h data found. Run fetch_futures_data.py first.")


def _fmt(v: float) -> str:
    if v != v:
        return "n/a"
    if abs(v) < 10:
        return f"{v:.4f}"
    return f"{v:,.4f}"


def correlation_analysis(btc: pd.DataFrame, spy: pd.DataFrame) -> None:
    """Correlation between SPY 1h return and BTC next-1h return during US hours."""
    # Merge SPY onto BTC by close_time using backward asof (strictly before).
    btc_s = btc[["close_time"]].reset_index(drop=True).copy()
    merged = pd.merge_asof(
        btc_s,
        spy[["close_time", "spy_ret_1h", "spy_ret_3h"]],
        on="close_time",
        direction="backward",
        allow_exact_matches=False,
    )

    df = pd.concat(
        [btc.reset_index(drop=True), merged[["spy_ret_1h", "spy_ret_3h"]]],
        axis=1,
    )
    df["btc_ret_1h"] = np.log(df["close"] / df["close"].shift(1))
    df["btc_next_1h"] = df["btc_ret_1h"].shift(-1)

    # Restrict to US market hours (13-20 UTC).
    df["hour"] = df["open_time"].dt.hour
    us = df[(df["hour"] >= 13) & (df["hour"] < 20)].copy()

    print(f"\n=== Correlation analysis (US hours only) ===")
    print(f"  Observations in US hours: {len(us)}")

    for col, label in [("spy_ret_1h", "SPY 1h"), ("spy_ret_3h", "SPY 3h")]:
        s = us.dropna(subset=[col, "btc_next_1h"])
        if len(s) < 30:
            print(f"  {label}: insufficient data ({len(s)} rows)")
            continue
        corr = s[col].corr(s["btc_next_1h"])
        n = len(s)
        # t-statistic for correlation coefficient.
        t_stat = corr * np.sqrt((n - 2) / max(1 - corr**2, 1e-12))
        print(f"  {label} → next-BTC-1h correlation: {corr:+.4f}  (n={n}, t={t_stat:+.2f})")
        # Bonferroni threshold for 2 tests: |t| > 2.24
        if abs(t_stat) > 2.24:
            print(f"    -> passes Bonferroni threshold (|t| > 2.24)")
        else:
            print(f"    -> fails Bonferroni threshold (|t| > 2.24). Likely noise.")


def simple_strategy(btc: pd.DataFrame, spy: pd.DataFrame, entry_thresh_bps: float = 0.0) -> pd.Series:
    """Target: long BTC when most recent SPY 3h return > threshold, during US hours."""
    btc_s = btc[["close_time"]].reset_index(drop=True).copy()
    merged = pd.merge_asof(
        btc_s,
        spy[["close_time", "spy_ret_3h"]],
        on="close_time",
        direction="backward",
        allow_exact_matches=False,
    )
    spy_sig = (merged["spy_ret_3h"] > entry_thresh_bps / 10_000.0).astype(float).fillna(0.0)
    in_us = btc["open_time"].dt.hour.between(13, 19).reset_index(drop=True).astype(float)

    signal = (spy_sig * in_us).fillna(0.0)
    target = signal.shift(1).fillna(0.0)
    return target


def main() -> None:
    spy = fetch_spy()
    spy["close_time"] = spy["close_time"].astype("datetime64[us, UTC]")
    btc = load_btc()

    print(f"\n=== Data overlap check ===")
    print(f"  SPY: {spy['open_time'].iloc[0]} -> {spy['open_time'].iloc[-1]}  ({len(spy)} bars)")
    print(f"  BTC: {btc['open_time'].iloc[0]} -> {btc['open_time'].iloc[-1]}  ({len(btc)} bars)")

    # Restrict BTC to SPY's date range.
    btc = btc[
        (btc["open_time"] >= spy["open_time"].iloc[0]) &
        (btc["open_time"] <= spy["open_time"].iloc[-1])
    ].reset_index(drop=True)
    print(f"  BTC after alignment: {len(btc)} bars")

        # Normalize timestamp precision so merge_asof works on both frames.
    spy["close_time"] = spy["close_time"].astype("datetime64[us, UTC]")
    btc["close_time"] = btc["close_time"].astype("datetime64[us, UTC]")

    if len(btc) < 500:
        logger.error("Not enough overlapping BTC bars. Extend BTC fetch.")
        return

    # ---- Part 1: Correlation ----
    correlation_analysis(btc, spy)

    # ---- Part 2: Backtest (best-effort) ----
    print(f"\n=== Simple strategy backtest ===")
    target = simple_strategy(btc, spy, entry_thresh_bps=0.0).reset_index(drop=True)
    target.index = btc.index

    result = run_backtest(
        df=btc,
        target_position=target,
        cost_model=COST,
        initial_equity=INITIAL_EQUITY,
        periods_per_year=PERIODS_PER_YEAR,
    )

    pos = result.position
    n_entries = int(((pos.diff() == 1.0).sum()))
    years = max((btc["open_time"].iloc[-1] - btc["open_time"].iloc[0]).days / 365.25, 0.01)

    print(f"  Final equity:      {result.final_equity:.4f}  (start {INITIAL_EQUITY:.2f})")
    print(f"  Entries:           {n_entries}  ({n_entries / years:.0f}/year)")
    for k, v in result.metrics.items():
        if isinstance(v, float):
            print(f"  {k:16s}   {_fmt(v)}")
        else:
            print(f"  {k:16s}   {v}")

    print(f"\n  Cost: {FUTURES_TAKER_BPS:.1f} bps/side, {2 * FUTURES_TAKER_BPS:.1f} bps round trip")


if __name__ == "__main__":
    main()

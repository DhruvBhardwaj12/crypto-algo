"""Backtest RSI(2) mean reversion on 5m BTC/ETH futures.

Run:
    uv run python scripts/backtest_rsi2_intraday.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from loguru import logger

from crypto_algo.backtesting.simulator import buy_and_hold_equity, run_backtest
from crypto_algo.data.loaders import load_parquet
from crypto_algo.strategies.rsi2_intraday import rsi2_intraday_target

START = "2023-01-01"
END = "2025-01-01"
SYMBOLS = ["BTCUSDT", "ETHUSDT"]
INITIAL_EQUITY = 10_000.0
FUTURES_TAKER_BPS = 5.0
PERIODS_PER_YEAR = 12 * 24 * 365  # 5m bars


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


def _load(symbol: str):
    return load_parquet(
        Path(f"data/raw/binance/futures/{symbol}/5m/{START}_{END}.parquet")
    )


def _fmt(v: float) -> str:
    if v != v:
        return "n/a"
    if abs(v) < 10:
        return f"{v:.4f}"
    return f"{v:,.4f}"


def main() -> None:
    results = {}
    for symbol in SYMBOLS:
        df = _load(symbol)

        target = rsi2_intraday_target(df).reset_index(drop=True)
        target.index = df.index

        result = run_backtest(
            df=df,
            target_position=target,
            cost_model=COST,
            initial_equity=INITIAL_EQUITY,
            periods_per_year=PERIODS_PER_YEAR,
        )
        results[symbol] = result

        bh = buy_and_hold_equity(df, INITIAL_EQUITY)
        bh_ret = float(bh.iloc[-1] / bh.iloc[0] - 1.0)

        pos = result.position
        n_entries = int(((pos.diff() == 1.0).sum()))
        n_exits = int(((pos.diff() == -1.0).sum()))
        years = (df["open_time"].iloc[-1] - df["open_time"].iloc[0]).days / 365.25

        print(f"\n=== {symbol} — RSI(2) 5m mean reversion ===")
        print(f"Rows: {len(df)}  range: {df['open_time'].iloc[0]} -> {df['open_time'].iloc[-1]}")
        print(f"Years covered: {years:.2f}")
        print(f"Final equity: {result.final_equity:,.2f}  (start {INITIAL_EQUITY:,.2f})")
        print(f"Buy & hold final: {bh.iloc[-1]:,.2f}  (return {bh_ret:+.2%})")
        print(f"Entries: {n_entries}  Exits: {n_exits}")
        print(f"Trades/year: {n_entries / years:.0f}")
        print(f"Trades/day (approx): {n_entries / (years * 365.25):.2f}")
        print("Metrics:")
        for k, v in result.metrics.items():
            if isinstance(v, float):
                print(f"  {k:16s} {_fmt(v)}")
            else:
                print(f"  {k:16s} {v}")

    print()
    print(f"Cost: {FUTURES_TAKER_BPS:.1f} bps/side, {2 * FUTURES_TAKER_BPS:.1f} bps round trip")


if __name__ == "__main__":
    main()
    
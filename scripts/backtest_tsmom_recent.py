"""TSMOM recency test: frozen params on 2025-2026 data.

Frozen params from LAB-001 walk-forward:
  BTC: lookback=168
  ETH: lookback=84

Run:
    uv run python scripts/backtest_tsmom_recent.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from loguru import logger

from crypto_algo.backtesting.simulator import run_backtest
from crypto_algo.data.loaders import load_parquet
from crypto_algo.strategies.tsmom import tsmom_target

START = "2025-01-01"
END = "2026-09-23"
INITIAL_EQUITY = 10_000.0
PERIODS_PER_YEAR = 6 * 365

# Frozen params. Do not adjust.
LOOKBACK_BY_SYMBOL = {"BTCUSDT": 168, "ETHUSDT": 84}


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


def _fmt(v: float) -> str:
    if v != v:
        return "n/a"
    if abs(v) < 10:
        return f"{v:+.4f}"
    return f"{v:+,.2f}"


def main() -> None:
    print(f"\n{'=' * 78}")
    print(f"  TSMOM — Recent OOS test ({START} to {END})")
    print(f"  Frozen params from LAB-001 walk-forward")
    print(f"{'=' * 78}")

    for symbol in ["BTCUSDT", "ETHUSDT"]:
        path = Path(f"data/raw/binance/futures/{symbol}/4h/{START}_{END}.parquet")
        if not path.exists():
            logger.error("Missing data for {}", symbol)
            continue

        df = load_parquet(path)
        lookback = LOOKBACK_BY_SYMBOL[symbol]

        print(f"\n  --- {symbol} (lookback={lookback}) ---")
        print(f"  Bars: {len(df)}  ({df['open_time'].iloc[0].date()} to {df['open_time'].iloc[-1].date()})")

        target = tsmom_target(df["close"], lookback_n=lookback).reset_index(drop=True)
        target.index = df.index

        print(f"  {'bps/side':>9s}  {'RT bps':>7s}  {'Final$':>12s}  {'TotalRet':>10s}  "
              f"{'CAGR':>8s}  {'Sharpe':>8s}  {'MaxDD':>9s}  {'Trades':>7s}")

        for bps in [5.0, 10.0, 15.0]:
            result = run_backtest(
                df=df, target_position=target, cost_model=_Cost(bps),
                initial_equity=INITIAL_EQUITY, periods_per_year=PERIODS_PER_YEAR,
            )
            m = result.metrics
            print(
                f"  {bps:>9.1f}  {2 * bps:>7.1f}  "
                f"{result.final_equity:>12,.0f}  "
                f"{m['total_return']:>+9.1%}  "
                f"{m['cagr']:>+7.1%}  "
                f"{m['sharpe']:>+8.2f}  "
                f"{m['max_drawdown']:>+9.2%}  "
                f"{int(m['n_trades']):>7d}"
            )


if __name__ == "__main__":
    main()
    
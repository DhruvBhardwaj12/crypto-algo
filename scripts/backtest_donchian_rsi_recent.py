"""EXP-004 recency test: frozen params on 2025-2026 data.

Frozen params from EXP-004 validation:
  dc=55, dc_exit=20, rsi4h_entry=52, rsi4h_exit=48, rsi1h_entry=50

Run:
    uv run python scripts/backtest_donchian_rsi_recent.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from loguru import logger

from crypto_algo.backtesting.simulator import run_backtest
from crypto_algo.data.loaders import load_parquet
from crypto_algo.strategies.donchian_rsi import (
    donchian_rsi_target,
    tag_with_4h_regime,
)

START = "2025-01-01"
END = "2026-09-23"
INITIAL_EQUITY = 10_000.0
PERIODS_PER_YEAR = 24 * 365

# Frozen from EXP-004 validation.
DC = 55
DC_EXIT = 20
RSI_1H_PERIOD = 14
RSI_1H_ENTRY = 50.0
RSI_4H_PERIOD = 14
RSI_4H_ENTRY = 52.0
RSI_4H_EXIT = 48.0


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
    print(f"  EXP-004 (Donchian + RSI) — Recent OOS test ({START} to {END})")
    print(f"  Frozen params from EXP-004 validation")
    print(f"{'=' * 78}")

    for symbol in ["BTCUSDT", "ETHUSDT"]:
        path_1h = Path(f"data/raw/binance/futures/{symbol}/1h/{START}_{END}.parquet")
        path_4h = Path(f"data/raw/binance/futures/{symbol}/4h/{START}_{END}.parquet")
        if not path_1h.exists() or not path_4h.exists():
            logger.error("Missing data for {} (1h={}, 4h={})",
                         symbol, path_1h.exists(), path_4h.exists())
            continue

        df_1h = load_parquet(path_1h)
        df_4h = load_parquet(path_4h)

        print(f"\n  --- {symbol} ---")
        print(f"  1H bars: {len(df_1h)}  4H bars: {len(df_4h)}")
        print(f"  Range: {df_1h['open_time'].iloc[0].date()} to {df_1h['open_time'].iloc[-1].date()}")

        rsi_4h = tag_with_4h_regime(df_1h, df_4h, rsi_period=RSI_4H_PERIOD)
        target = donchian_rsi_target(
            df_1h, rsi_4h,
            donchian_entry_lookback=DC,
            donchian_exit_lookback=DC_EXIT,
            rsi_1h_period=RSI_1H_PERIOD,
            rsi_1h_entry=RSI_1H_ENTRY,
            rsi_4h_entry=RSI_4H_ENTRY,
            rsi_4h_exit=RSI_4H_EXIT,
        ).reset_index(drop=True)
        target.index = df_1h.index

        print(f"  {'bps/side':>9s}  {'RT bps':>7s}  {'Final$':>12s}  {'TotalRet':>10s}  "
              f"{'CAGR':>8s}  {'Sharpe':>8s}  {'MaxDD':>9s}  {'Trades':>7s}")

        for bps in [5.0, 10.0, 15.0]:
            result = run_backtest(
                df=df_1h, target_position=target, cost_model=_Cost(bps),
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
    
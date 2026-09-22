"""Backtest Donchian + RSI multi-timeframe strategy on BTC/ETH futures.

Run:
    uv run python scripts/backtest_donchian_rsi.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from loguru import logger

from crypto_algo.backtesting.futures_costs import FuturesCostModel
from crypto_algo.backtesting.simulator import BacktestResult, buy_and_hold_equity, run_backtest
from crypto_algo.data.loaders import load_parquet
from crypto_algo.strategies.donchian_rsi import (
    donchian_rsi_target,
    tag_with_4h_regime,
)

START = "2020-01-01"
END = "2025-01-01"
SYMBOLS = ["BTCUSDT", "ETHUSDT"]
INITIAL_EQUITY = 10_000.0

# Futures cost model: buy+short perp leg is ~0.34% round trip.
# But our strategy is long-only unlevered futures, so only one leg.
# We use a simple futures cost of 5 bps per side (taker on entry/exit).
# This is generous — real fills may be better with limit orders.
FUTURES_TAKER_BPS = 5.0


class _FuturesSingleLegCost:
    """Minimal cost model: symmetric taker cost per side."""

    def __init__(self, cost_bps: float):
        self.cost_bps = cost_bps

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
        if side not in ("buy", "sell"):
            raise ValueError(side)
        return self.cost_bps / 10_000.0


COST = _FuturesSingleLegCost(FUTURES_TAKER_BPS)
# periods_per_year for 1H bars:
PERIODS_PER_YEAR = 24 * 365


def _load_1h(symbol: str) -> pd.DataFrame:
    return load_parquet(
        Path(f"data/raw/binance/futures/{symbol}/1h/{START}_{END}.parquet")
    )


def _load_4h(symbol: str) -> pd.DataFrame:
    return load_parquet(
        Path(f"data/raw/binance/futures/{symbol}/4h/{START}_{END}.parquet")
    )


def _fmt(v: float) -> str:
    if v != v:
        return "n/a"
    if abs(v) < 10:
        return f"{v:.4f}"
    return f"{v:,.4f}"


def main() -> None:
    results: dict[str, BacktestResult] = {}

    for symbol in SYMBOLS:
        df_1h = _load_1h(symbol)
        df_4h = _load_4h(symbol)

        rsi_4h = tag_with_4h_regime(df_1h, df_4h, rsi_period=14)

        target = donchian_rsi_target(df_1h, rsi_4h).reset_index(drop=True)
        target.index = df_1h.index

        result = run_backtest(
            df=df_1h,
            target_position=target,
            cost_model=COST,
            initial_equity=INITIAL_EQUITY,
            periods_per_year=PERIODS_PER_YEAR,
        )
        results[symbol] = result

        bh = buy_and_hold_equity(df_1h, INITIAL_EQUITY)
        bh_return = float(bh.iloc[-1] / bh.iloc[0] - 1.0)

        # Count entry/exit transitions.
        pos = result.position
        n_entries = int(((pos.diff() == 1.0).sum()))
        n_exits = int(((pos.diff() == -1.0).sum()))

        print(f"\n=== {symbol} — Donchian + RSI multi-timeframe (1H / 4H) ===")
        print(f"Rows: {len(df_1h)}  ({df_1h['open_time'].iloc[0]} -> {df_1h['open_time'].iloc[-1]})")
        print(f"Final equity: {result.final_equity:,.2f} (start {INITIAL_EQUITY:,.2f})")
        print(f"Buy & hold final: {bh.iloc[-1]:,.2f}  (return {bh_return:+.2%})")
        print(f"Entries: {n_entries}  Exits: {n_exits}")
        print("Metrics:")
        for k, v in result.metrics.items():
            if isinstance(v, float):
                print(f"  {k:16s} {_fmt(v)}")
            else:
                print(f"  {k:16s} {v}")

    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    for symbol, res in results.items():
        ax.plot(res.equity.index, res.equity.values, label=f"{symbol} strategy")
    btc_1h = _load_1h("BTCUSDT")
    bh = buy_and_hold_equity(btc_1h, INITIAL_EQUITY)
    ax.plot(bh.index, bh.values, label="BTC B&H", linestyle="--", alpha=0.7)
    ax.set_title(f"Donchian + RSI multi-timeframe — {START} to {END}")
    ax.set_xlabel("Time (UTC)")
    ax.set_ylabel("Equity (USD)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = Path("research/backtest_donchian_rsi.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=120)
    logger.success("Saved plot to {}", out)

    print()
    print(f"Cost model: {FUTURES_TAKER_BPS:.1f} bps/side, "
          f"{2 * FUTURES_TAKER_BPS:.1f} bps round trip "
          f"= {2 * FUTURES_TAKER_BPS / 100:.4f}%")


if __name__ == "__main__":
    main()
    
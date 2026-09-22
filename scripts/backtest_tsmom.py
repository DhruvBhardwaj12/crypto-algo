"""Backtest TSMOM on 4H BTC/ETH futures.

Run:
    uv run python scripts/backtest_tsmom.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from loguru import logger

from crypto_algo.backtesting.simulator import buy_and_hold_equity, run_backtest
from crypto_algo.data.loaders import load_parquet
from crypto_algo.strategies.tsmom import tsmom_target

START = "2020-01-01"
END = "2025-01-01"
SYMBOLS = ["BTCUSDT", "ETHUSDT"]
INITIAL_EQUITY = 10_000.0
LOOKBACK = 42
FUTURES_TAKER_BPS = 5.0
PERIODS_PER_YEAR = 6 * 365  # 4H bars


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
        Path(f"data/raw/binance/futures/{symbol}/4h/{START}_{END}.parquet")
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
        target = tsmom_target(df["close"], lookback_n=LOOKBACK).reset_index(drop=True)
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

        print(f"\n=== {symbol} — TSMOM(lookback={LOOKBACK}) on 4H ===")
        print(f"Rows: {len(df)}  range: {df['open_time'].iloc[0]} -> {df['open_time'].iloc[-1]}")
        print(f"Final equity: {result.final_equity:,.2f}  (start {INITIAL_EQUITY:,.2f})")
        print(f"Buy & hold final: {bh.iloc[-1]:,.2f}  (return {bh_ret:+.2%})")
        print(f"Entries: {n_entries}  Exits: {n_exits}")
        print("Metrics:")
        for k, v in result.metrics.items():
            if isinstance(v, float):
                print(f"  {k:16s} {_fmt(v)}")
            else:
                print(f"  {k:16s} {v}")

    fig, ax = plt.subplots(figsize=(12, 6))
    for symbol, res in results.items():
        ax.plot(res.equity.index, res.equity.values, label=f"{symbol} TSMOM")
    btc = _load("BTCUSDT")
    bh = buy_and_hold_equity(btc, INITIAL_EQUITY)
    ax.plot(bh.index, bh.values, label="BTC B&H", linestyle="--", alpha=0.7)
    ax.set_title(f"TSMOM({LOOKBACK}) on 4H futures — {START} to {END}")
    ax.set_xlabel("Time (UTC)")
    ax.set_ylabel("Equity (USD)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = Path("research/backtest_tsmom.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=120)
    logger.success("Saved plot to {}", out)

    print()
    print(f"Cost: {FUTURES_TAKER_BPS:.1f} bps/side, {2 * FUTURES_TAKER_BPS:.1f} bps round trip")


if __name__ == "__main__":
    main()
    
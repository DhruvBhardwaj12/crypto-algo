"""LAB-006 — The Frequency Map.

Run the SAME signal (TSMOM, 1-week equivalent lookback) across every
timeframe we have data for, on a common 6-month window. Measure gross
and net Sharpe at each timeframe. Produce a Sharpe-vs-timeframe curve
that shows exactly where cost drag kills the signal.

Timeframes and lookbacks (1-week equivalent):
  1D  -> 7 bars
  4H  -> 42 bars
  1H  -> 168 bars
  15m -> 672 bars
  5m  -> 2016 bars
  1m  -> 10080 bars

Window: 2024-07-01 to 2025-01-01.
Cost: 0 bps (gross) and 10 bps round trip (net).

Run:
    uv run python scripts/lab006_frequency_map.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger

from crypto_algo.backtesting.simulator import run_backtest
from crypto_algo.data.loaders import load_parquet
from crypto_algo.strategies.tsmom import tsmom_target

START = "2024-07-01"
END = "2025-01-01"
SYMBOLS = ["BTCUSDT", "ETHUSDT"]
INITIAL_EQUITY = 10_000.0
NET_COST_BPS = 10.0  # round-trip cost for the "net" run

# Timeframe -> (filename_start, filename_end, lookback_bars, periods_per_year)
# We use the 2020-2025 files for 1d/4h/1h (already on disk) and 2024-07-01
# files for 15m/5m/1m.
TIMEFRAMES = [
    ("1d", "2020-01-01", "2025-01-01", 7, 365),
    ("4h", "2020-01-01", "2025-01-01", 42, 6 * 365),
    ("1h", "2020-01-01", "2025-01-01", 168, 24 * 365),
    ("15m", "2024-07-01", "2025-01-01", 672, 96 * 365),
    ("5m", "2024-07-01", "2025-01-01", 2016, 288 * 365),
    ("1m", "2024-07-01", "2025-01-01", 10080, 1440 * 365),
]


class _Cost:
    def __init__(self, bps_round_trip: float):
        # Split evenly: bps/2 per side.
        self.per_side_bps = bps_round_trip / 2.0

    @property
    def buy_cost_bps(self) -> float:
        return self.per_side_bps

    @property
    def sell_cost_bps(self) -> float:
        return self.per_side_bps

    @property
    def round_trip_bps(self) -> float:
        return 2 * self.per_side_bps

    def cost_fraction(self, side: str) -> float:
        return self.per_side_bps / 10_000.0


COST_GROSS = _Cost(0.0)
COST_NET = _Cost(NET_COST_BPS)


def _load(symbol: str, interval: str, file_start: str, file_end: str) -> pd.DataFrame | None:
    path = Path(f"data/raw/binance/futures/{symbol}/{interval}/{file_start}_{file_end}.parquet")
    if not path.exists():
        logger.warning("Missing: {}", path)
        return None
    df = load_parquet(path)
    # Restrict to the common window.
    lo = pd.Timestamp(START, tz="UTC")
    hi = pd.Timestamp(END, tz="UTC")
    df = df[(df["open_time"] >= lo) & (df["open_time"] < hi)].reset_index(drop=True)
    return df


def _run(df: pd.DataFrame, lookback: int, cost_model: _Cost, periods_per_year: int):
    target = tsmom_target(df["close"], lookback_n=lookback).reset_index(drop=True)
    target.index = df.index
    return run_backtest(
        df=df,
        target_position=target,
        cost_model=cost_model,
        initial_equity=INITIAL_EQUITY,
        periods_per_year=periods_per_year,
    )


def _edge_per_trade_bps(net_result, n_trades: int) -> float:
    """Implied gross edge per trade in bps."""
    if n_trades == 0:
        return float("nan")
    # total net return as a fraction
    total_net = net_result.final_equity / INITIAL_EQUITY - 1.0
    # Mean net return per trade.
    mean_net_per_trade = total_net / n_trades
    # Add back cost per trade.
    cost_per_trade = NET_COST_BPS / 10_000.0
    gross_edge_per_trade = mean_net_per_trade + cost_per_trade
    return gross_edge_per_trade * 10_000.0


def main() -> None:
    rows = []
    for symbol in SYMBOLS:
        print(f"\n{'=' * 90}")
        print(f"  {symbol}  —  LAB-006 frequency map")
        print(f"{'=' * 90}")
        print(f"  {'tf':>5s} {'bars':>8s} {'trades':>8s} {'trd/yr':>8s} "
              f"{'Sharpe_gross':>13s} {'Sharpe_net':>11s} {'edge/trd':>10s} "
              f"{'cost_drag%':>11s}")

        for interval, fs, fe, lookback, ppy in TIMEFRAMES:
            df = _load(symbol, interval, fs, fe)
            if df is None or len(df) < lookback + 100:
                print(f"  {interval:>5s}  SKIPPED (insufficient data)")
                continue

            # Gross (0 cost)
            r_gross = _run(df, lookback, COST_GROSS, ppy)
            # Net (10 bps round trip)
            r_net = _run(df, lookback, COST_NET, ppy)

            n_trades = int(r_net.metrics.get("n_trades", 0))
            days = max((df["open_time"].iloc[-1] - df["open_time"].iloc[0]).days, 1)
            trades_per_year = n_trades * 365.0 / days

            sharpe_gross = float(r_gross.metrics.get("sharpe", float("nan")))
            sharpe_net = float(r_net.metrics.get("sharpe", float("nan")))

            edge_bps = _edge_per_trade_bps(r_net, n_trades)

            # Cost drag as annualized % of equity.
            cost_drag_pct = trades_per_year * NET_COST_BPS / 100.0

            print(f"  {interval:>5s} {len(df):>8d} {n_trades:>8d} "
                  f"{trades_per_year:>8.0f} {sharpe_gross:>13.3f} "
                  f"{sharpe_net:>11.3f} {edge_bps:>10.2f} "
                  f"{cost_drag_pct:>11.1f}")

            rows.append({
                "symbol": symbol,
                "timeframe": interval,
                "bars": len(df),
                "lookback": lookback,
                "n_trades": n_trades,
                "trades_per_year": trades_per_year,
                "sharpe_gross": sharpe_gross,
                "sharpe_net": sharpe_net,
                "edge_per_trade_bps": edge_bps,
                "cost_drag_pct": cost_drag_pct,
            })

    df_rows = pd.DataFrame(rows)
    df_rows.to_csv("research/lab006_frequency_map.csv", index=False)
    logger.success("Saved research/lab006_frequency_map.csv")

    # ---- Plot ----
    # Order timeframes from slowest to fastest for the x-axis.
    order = ["1d", "4h", "1h", "15m", "5m", "1m"]

    fig, ax = plt.subplots(figsize=(11, 6))

    for symbol in SYMBOLS:
        sub = df_rows[df_rows["symbol"] == symbol].set_index("timeframe")
        sub = sub.reindex([t for t in order if t in sub.index])
        x = np.arange(len(sub))
        ax.plot(x, sub["sharpe_gross"], marker="o", linestyle="-",
                label=f"{symbol} gross (0 bps)")
        ax.plot(x, sub["sharpe_net"], marker="s", linestyle="--",
                label=f"{symbol} net (10 bps)")

    ax.axhline(0, color="black", linewidth=0.8, alpha=0.5)
    ax.set_xticks(np.arange(len(order)))
    ax.set_xticklabels(order)
    ax.set_xlabel("Timeframe (slower → faster)")
    ax.set_ylabel("Sharpe ratio")
    ax.set_title(f"LAB-006 — TSMOM Sharpe vs timeframe — {START} to {END}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("research/lab006_frequency_map.png", dpi=120)
    plt.close(fig)
    logger.success("Saved research/lab006_frequency_map.png")


if __name__ == "__main__":
    main()
    
"""Backtest cross-sectional momentum on the RECENT out-of-sample period.

Uses params chosen by 2020-2025 walk-forward (84/3/2/6). No re-optimization.
Tests the same universe on 2025-01-01 to present.

If the strategy is decaying, this will show it. If it's still working, we
have genuine fresh evidence.

Run:
    uv run python scripts/backtest_cs_recent.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import yaml
from loguru import logger

from crypto_algo.backtesting.portfolio_simulator import run_portfolio_backtest
from crypto_algo.data.loaders import load_parquet
from crypto_algo.strategies.cross_sectional_momentum import (
    cross_sectional_momentum_weights,
)

START = "2025-01-01"
END = "2026-09-23"
INITIAL_EQUITY = 10_000.0
PERIODS_PER_YEAR = 6 * 365

# Params chosen by 2020-2025 walk-forward. FROZEN. Do not adjust.
LOOKBACK = 84
N_LONG = 3
N_SHORT = 2
REBALANCE = 6


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


def _load_all(symbols: list[str]) -> pd.DataFrame:
    series = {}
    for symbol in symbols:
        path = Path(f"data/raw/binance/futures/{symbol}/4h/{START}_{END}.parquet")
        if not path.exists():
            logger.warning("Missing: {}", path)
            continue
        df = load_parquet(path).sort_values("open_time").set_index("open_time")
        s = df["close"].rename(symbol)
        if len(s) > 100:
            series[symbol] = s
    if not series:
        raise ValueError("No symbols loaded.")
    wide = pd.DataFrame(series).sort_index()
    return wide


def _run(prices: pd.DataFrame, cost_bps: float) -> dict:
    weights = cross_sectional_momentum_weights(
        prices,
        lookback=LOOKBACK,
        n_long=N_LONG,
        n_short=N_SHORT,
        rebalance_every=REBALANCE,
    )
    result = run_portfolio_backtest(
        prices=prices, weights=weights, cost_model=_Cost(cost_bps),
        initial_equity=INITIAL_EQUITY, periods_per_year=PERIODS_PER_YEAR,
    )
    return result


def _fmt(v: float) -> str:
    if v != v:
        return "n/a"
    if abs(v) < 10:
        return f"{v:+.4f}"
    return f"{v:+,.2f}"


def main() -> None:
    with open("config/settings.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)["data"]
    prices = _load_all(cfg["symbols"])

    non_nan = prices.notna().sum(axis=1)
    valid_start = non_nan[non_nan >= (N_LONG + N_SHORT)].index.min()
    prices = prices.loc[prices.index >= valid_start]

    days = (prices.index[-1] - prices.index[0]).days
    years = days / 365.25

    print(f"\n{'=' * 78}")
    print(f"  Cross-Sectional Momentum — RECENT OOS TEST")
    print(f"  Frozen params: lookback={LOOKBACK}, n_long={N_LONG}, n_short={N_SHORT}, rebalance={REBALANCE}")
    print(f"{'=' * 78}")
    print(f"  Universe: {prices.shape[1]} symbols")
    print(f"  Window:   {prices.index[0]} -> {prices.index[-1]}  ({years:.2f} years)")
    print(f"  Bars:     {len(prices)}")
    print()

    print(f"  {'bps/side':>9s}  {'RT bps':>7s}  {'Final$':>12s}  {'TotalRet':>10s}  "
          f"{'CAGR':>8s}  {'Sharpe':>8s}  {'MaxDD':>9s}  {'Calmar':>8s}")

    for bps in [5.0, 10.0, 15.0, 25.0]:
        result = _run(prices, cost_bps=bps)
        m = result.metrics
        print(
            f"  {bps:>9.1f}  {2 * bps:>7.1f}  "
            f"{result.equity.iloc[-1]:>12,.0f}  "
            f"{m['total_return']:>+9.1%}  "
            f"{m['cagr']:>+7.1%}  "
            f"{m['sharpe']:>+8.2f}  "
            f"{m['max_drawdown']:>+9.2%}  "
            f"{m['calmar']:>+8.2f}"
        )

    # Plot equity curve at 5 bps (base case)
    result = _run(prices, cost_bps=5.0)
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(result.equity.index, result.equity.values)
    ax.set_title(f"Cross-Sectional Momentum — Recent OOS period ({prices.index[0].date()} to {prices.index[-1].date()})")
    ax.set_xlabel("Date (UTC)")
    ax.set_ylabel("Equity (USD)")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("research/backtest_cs_recent.png", dpi=120)
    logger.success("Saved plot to research/backtest_cs_recent.png")


if __name__ == "__main__":
    main()
    
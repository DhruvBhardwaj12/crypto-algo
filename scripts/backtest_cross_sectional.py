"""Backtest cross-sectional momentum on 20 crypto futures.

Run:
    uv run python scripts/backtest_cross_sectional.py
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

START = "2020-01-01"
END = "2025-01-01"
INITIAL_EQUITY = 10_000.0
FUTURES_TAKER_BPS = 5.0
PERIODS_PER_YEAR = 6 * 365

# Strategy parameters
LOOKBACK = 42       # 7 days of 4H bars
N_LONG = 3
N_SHORT = 3
REBALANCE_EVERY = 6  # 1 day


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


def _load_all(symbols: list[str]) -> pd.DataFrame:
    """Load each symbol's 4H close price. Return wide DataFrame."""
    series = {}
    for symbol in symbols:
        path = Path(f"data/raw/binance/futures/{symbol}/4h/{START}_{END}.parquet")
        if not path.exists():
            logger.warning("Missing {} — skipping", path)
            continue
        df = load_parquet(path)
        df = df.sort_values("open_time").set_index("open_time")
        s = df["close"].rename(symbol)
        if len(s) > 100:
            series[symbol] = s
            logger.info("Loaded {}: {} bars", symbol, len(s))

    if not series:
        raise ValueError("No symbols loaded.")

    # Align on union of all timestamps; missing values become NaN.
    wide = pd.DataFrame(series)
    wide = wide.sort_index()
    logger.success("Wide price matrix: {} rows x {} cols", *wide.shape)
    return wide


def _fmt(v: float) -> str:
    if v != v:
        return "n/a"
    if abs(v) < 10:
        return f"{v:.4f}"
    return f"{v:,.4f}"


def main() -> None:
    with open("config/settings.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)["data"]
    symbols = cfg["symbols"]

    logger.info("Loading {} symbols from {}", len(symbols), cfg["futures_base_url"])

    prices = _load_all(symbols)

    # Only start backtest from a point where at least a few symbols have data.
    # Drop very early rows (before enough symbols exist).
    non_nan_counts = prices.notna().sum(axis=1)
    min_universe = N_LONG + N_SHORT
    valid_start = non_nan_counts[non_nan_counts >= min_universe].index.min()
    if valid_start is not None and valid_start > prices.index[0]:
        logger.info("Backtest start: {} (first bar with {} symbols)",
                    valid_start, min_universe)
        prices = prices.loc[prices.index >= valid_start]

    logger.info("Backtest window: {} to {}", prices.index[0], prices.index[-1])

    # Build weights.
    weights = cross_sectional_momentum_weights(
        prices,
        lookback=LOOKBACK,
        n_long=N_LONG,
        n_short=N_SHORT,
        rebalance_every=REBALANCE_EVERY,
    )

    # Run backtest.
    result = run_portfolio_backtest(
        prices=prices,
        weights=weights,
        cost_model=COST,
        initial_equity=INITIAL_EQUITY,
        periods_per_year=PERIODS_PER_YEAR,
    )

    # Per-bar turnover in weight terms.
    delta = result.weights.diff().abs().sum(axis=1)
    mean_turnover = float(delta.mean())
    annual_turnover = mean_turnover * PERIODS_PER_YEAR

    # Number of bars per day (4H bars).
    bars_per_day = 6
    days = len(prices) / bars_per_day
    years = days / 365.25

    print(f"\n=== Cross-Sectional Momentum (LAB-007) ===")
    print(f"Symbols:         {list(prices.columns)}")
    print(f"Window:          {prices.index[0]} -> {prices.index[-1]}")
    print(f"Bars:            {len(prices)}  (~{years:.2f} years)")
    print(f"Universe size:   {N_LONG + N_SHORT} of {prices.shape[1]} ranked each rebalance")
    print(f"Lookback:        {LOOKBACK} bars (~{LOOKBACK / 6:.1f} days)")
    print(f"Rebalance:       every {REBALANCE_EVERY} bars (~{REBALANCE_EVERY / 6:.1f} days)")
    print(f"Cost:            {FUTURES_TAKER_BPS:.1f} bps/side, {2 * FUTURES_TAKER_BPS:.1f} bps round trip")
    print()
    print(f"Final equity:    {result.equity.iloc[-1]:,.2f}  (start {INITIAL_EQUITY:,.2f})")
    print(f"Mean daily turnover (weight units): {mean_turnover:.4f}")
    print(f"Annualized turnover:                {annual_turnover:.2f}")
    print("Metrics:")
    for k, v in result.metrics.items():
        if isinstance(v, float):
            print(f"  {k:16s} {_fmt(v)}")
        else:
            print(f"  {k:16s} {v}")

    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(result.equity.index, result.equity.values, label="Cross-sectional momentum")
    ax.set_title(f"LAB-007: Cross-sectional momentum — {prices.index[0].date()} to {prices.index[-1].date()}")
    ax.set_xlabel("Time (UTC)")
    ax.set_ylabel("Equity (USD)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = Path("research/backtest_cross_sectional.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=120)
    logger.success("Saved plot to {}", out)


if __name__ == "__main__":
    main()
    
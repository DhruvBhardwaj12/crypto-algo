"""LAB-007 concern checks.

Two targeted tests to attack the biggest vulnerabilities in the
cross-sectional momentum result:

  1. Survivorship bias: fix the universe as of specific historical
     dates, exclude any symbol that didn't exist then.

  2. Cost sensitivity: re-run at 5, 10, 15, 25 bps per side.

If the strategy holds up under both, the Sharpe 1.67 is robust.

Run:
    uv run python scripts/lab007_concern_checks.py
"""

from __future__ import annotations

from pathlib import Path

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
PERIODS_PER_YEAR = 6 * 365

LOOKBACK = 42
N_LONG = 3
N_SHORT = 3
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
        prices, lookback=LOOKBACK, n_long=N_LONG, n_short=N_SHORT,
        rebalance_every=REBALANCE,
    )
    result = run_portfolio_backtest(
        prices=prices, weights=weights, cost_model=_Cost(cost_bps),
        initial_equity=INITIAL_EQUITY, periods_per_year=PERIODS_PER_YEAR,
    )
    return {
        "final_equity": float(result.equity.iloc[-1]),
        "total_return": float(result.metrics.get("total_return", float("nan"))),
        "sharpe": float(result.metrics.get("sharpe", float("nan"))),
        "max_dd": float(result.metrics.get("max_drawdown", float("nan"))),
        "calmar": float(result.metrics.get("calmar", float("nan"))),
    }


def _fmt(v: float) -> str:
    if v != v:
        return "n/a"
    if abs(v) < 10:
        return f"{v:+.4f}"
    return f"{v:+,.2f}"


def _symbols_available_by(prices: pd.DataFrame, asof: pd.Timestamp, min_bars: int = 500) -> list[str]:
    """Return symbols with at least `min_bars` valid rows before `asof`."""
    cutoff = prices.loc[prices.index < asof]
    counts = cutoff.notna().sum()
    return [s for s in prices.columns if counts[s] >= min_bars]


def survivorship_test(prices: pd.DataFrame) -> None:
    print(f"\n{'=' * 78}")
    print(f"  Survivorship test — fix universe as of specific dates")
    print(f"{'=' * 78}")
    print(f"  {'asof':>12s}  {'n_symbols':>10s}  {'Sharpe':>9s}  {'Ret':>12s}  {'MaxDD':>10s}  symbols")

    for asof_str in ["2020-06-01", "2021-06-01", "2022-06-01", "2023-06-01"]:
        asof = pd.Timestamp(asof_str, tz="UTC")
        available = _symbols_available_by(prices, asof)
        if len(available) < N_LONG + N_SHORT:
            print(f"  {asof_str:>12s}  {len(available):>10d}  insufficient symbols")
            continue

        sub = prices[available]
        # Start backtest from `asof` onward.
        sub = sub.loc[sub.index >= asof].copy()
        # Need enough data.
        if len(sub) < 500:
            continue

        r = _run(sub, cost_bps=5.0)
        sym_short = ",".join(s.replace("USDT", "") for s in available[:8])
        if len(available) > 8:
            sym_short += f"... (+{len(available)-8})"
        print(
            f"  {asof_str:>12s}  {len(available):>10d}  "
            f"{r['sharpe']:>+9.3f}  {r['total_return']:>+11.1%}  "
            f"{r['max_dd']:>+10.2%}  {sym_short}"
        )


def cost_sensitivity_test(prices: pd.DataFrame) -> None:
    print(f"\n{'=' * 78}")
    print(f"  Cost sensitivity — re-run full backtest at different cost levels")
    print(f"{'=' * 78}")
    print(f"  {'bps/side':>9s}  {'RT bps':>8s}  {'Sharpe':>9s}  {'Ret':>12s}  {'MaxDD':>10s}  {'Calmar':>9s}")

    for bps in [0.0, 5.0, 10.0, 15.0, 25.0, 50.0]:
        r = _run(prices, cost_bps=bps)
        print(
            f"  {bps:>9.1f}  {2 * bps:>8.1f}  "
            f"{r['sharpe']:>+9.3f}  {r['total_return']:>+11.1%}  "
            f"{r['max_dd']:>+10.2%}  {r['calmar']:>+9.3f}"
        )


def main() -> None:
    with open("config/settings.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)["data"]

    prices = _load_all(cfg["symbols"])

    # Trim to shared date range.
    non_nan = prices.notna().sum(axis=1)
    valid_start = non_nan[non_nan >= (N_LONG + N_SHORT)].index.min()
    prices = prices.loc[prices.index >= valid_start]

    logger.info("Full universe: {} symbols, {} bars", prices.shape[1], prices.shape[0])

    # ---- Test 1: Survivorship ----
    survivorship_test(prices)

    # ---- Test 2: Cost sensitivity ----
    cost_sensitivity_test(prices)


if __name__ == "__main__":
    main()
    
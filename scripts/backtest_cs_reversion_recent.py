"""Cross-sectional mean reversion — recent OOS test.

Runs on 2025-01-01 to 2026-09-23 (fresh data, never validated on).
Also runs on 2020-2025 for comparison.

Run:
    uv run python scripts/backtest_cs_reversion_recent.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml
from loguru import logger

from crypto_algo.backtesting.portfolio_simulator import run_portfolio_backtest
from crypto_algo.data.loaders import load_parquet
from crypto_algo.strategies.cross_sectional_reversion import (
    cross_sectional_reversion_weights,
)

INITIAL_EQUITY = 10_000.0
PERIODS_PER_YEAR = 6 * 365
COST_BPS = 5.0

# We'll test a small grid of parameters, but NOT tune.
PARAM_GRID = [
    {"lookback": 42, "n_long": 3, "n_short": 3, "rebalance": 6},   # same as LAB-007
    {"lookback": 42, "n_long": 5, "n_short": 5, "rebalance": 6},
    {"lookback": 84, "n_long": 3, "n_short": 3, "rebalance": 6},
    {"lookback": 21, "n_long": 3, "n_short": 3, "rebalance": 3},
]


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


def _load_all(symbols: list[str], start: str, end: str) -> pd.DataFrame:
    series = {}
    for symbol in symbols:
        path = Path(f"data/raw/binance/futures/{symbol}/4h/{start}_{end}.parquet")
        if not path.exists():
            continue
        df = load_parquet(path).sort_values("open_time").set_index("open_time")
        s = df["close"].rename(symbol)
        if len(s) > 100:
            series[symbol] = s
    if not series:
        raise ValueError(f"No data for {start} to {end}.")
    wide = pd.DataFrame(series).sort_index()
    non_nan = wide.notna().sum(axis=1)
    valid_start = non_nan[non_nan >= 6].index.min()
    return wide.loc[wide.index >= valid_start]


def _run(prices: pd.DataFrame, params: dict) -> dict:
    weights = cross_sectional_reversion_weights(
        prices,
        lookback=int(params["lookback"]),
        n_long=int(params["n_long"]),
        n_short=int(params["n_short"]),
        rebalance_every=int(params["rebalance"]),
    )
    result = run_portfolio_backtest(
        prices=prices, weights=weights, cost_model=_Cost(COST_BPS),
        initial_equity=INITIAL_EQUITY, periods_per_year=PERIODS_PER_YEAR,
    )
    return {
        "final_equity": float(result.equity.iloc[-1]),
        "total_return": float(result.metrics.get("total_return", float("nan"))),
        "cagr": float(result.metrics.get("cagr", float("nan"))),
        "sharpe": float(result.metrics.get("sharpe", float("nan"))),
        "max_dd": float(result.metrics.get("max_drawdown", float("nan"))),
        "calmar": float(result.metrics.get("calmar", float("nan"))),
    }


def _fmt(v: float) -> str:
    if v != v:
        return "n/a"
    return f"{v:+,.3f}" if abs(v) < 10 else f"{v:+,.1f}"


def main() -> None:
    with open("config/settings.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)["data"]
    symbols = cfg["symbols"]

    for label, start, end in [
        ("HISTORICAL 2020-2025", "2020-01-01", "2025-01-01"),
        ("RECENT 2025-2026",     "2025-01-01", "2026-09-23"),
    ]:
        print(f"\n{'=' * 92}")
        print(f"  Cross-Sectional MEAN REVERSION — {label}")
        print(f"{'=' * 92}")

        try:
            prices = _load_all(symbols, start, end)
        except ValueError as e:
            print(f"  Skipped: {e}")
            continue

        days = (prices.index[-1] - prices.index[0]).days
        years = days / 365.25
        print(f"  Universe: {prices.shape[1]} symbols | Bars: {prices.shape[0]} | ~{years:.2f} years")
        print()
        print(f"  {'params':>20s}  {'Final$':>10s}  {'TotalRet':>10s}  {'CAGR':>8s}  {'Sharpe':>8s}  {'MaxDD':>9s}")

        for params in PARAM_GRID:
            try:
                r = _run(prices, params)
                tag = f"{params['lookback']}/{params['n_long']}/{params['n_short']}/{params['rebalance']}"
                print(
                    f"  {tag:>20s}  {r['final_equity']:>10,.0f}  "
                    f"{r['total_return']:>+9.1%}  {r['cagr']:>+7.1%}  "
                    f"{r['sharpe']:>+8.2f}  {r['max_dd']:>+9.2%}"
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Params {} failed: {}", params, exc)


if __name__ == "__main__":
    main()
    
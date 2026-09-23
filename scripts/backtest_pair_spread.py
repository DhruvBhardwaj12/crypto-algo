"""Backtest BTC/ETH pair spread mean reversion.

Runs on both historical (2020-2025) and recent (2025-2026) windows.
Tests a small parameter grid to check robustness.

Run:
    uv run python scripts/backtest_pair_spread.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from loguru import logger

from crypto_algo.backtesting.portfolio_simulator import run_portfolio_backtest
from crypto_algo.data.loaders import load_parquet
from crypto_algo.strategies.pair_spread import pair_spread_weights

INITIAL_EQUITY = 10_000.0
PERIODS_PER_YEAR = 6 * 365
COST_BPS = 5.0

GRID = [
    {"lookback": 21, "entry_z": 2.0, "exit_z": 0.5},
    {"lookback": 42, "entry_z": 2.0, "exit_z": 0.5},
    {"lookback": 42, "entry_z": 1.5, "exit_z": 0.5},
    {"lookback": 84, "entry_z": 2.0, "exit_z": 0.5},
    {"lookback": 42, "entry_z": 2.5, "exit_z": 1.0},
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


def _load_pair(start: str, end: str) -> pd.DataFrame:
    btc = load_parquet(
        Path(f"data/raw/binance/futures/BTCUSDT/4h/{start}_{end}.parquet")
    ).set_index("open_time")["close"].rename("BTCUSDT")
    eth = load_parquet(
        Path(f"data/raw/binance/futures/ETHUSDT/4h/{start}_{end}.parquet")
    ).set_index("open_time")["close"].rename("ETHUSDT")
    df = pd.concat([btc, eth], axis=1).dropna()
    return df


def _run(prices: pd.DataFrame, params: dict, cost_bps: float = COST_BPS):
    weights = pair_spread_weights(
        prices["BTCUSDT"],
        prices["ETHUSDT"],
        lookback=int(params["lookback"]),
        entry_z=float(params["entry_z"]),
        exit_z=float(params["exit_z"]),
        leg_weight=0.5,
    )
    # Align indices.
    weights = weights.reindex(prices.index).fillna(0.0)
    result = run_portfolio_backtest(
        prices=prices,
        weights=weights,
        cost_model=_Cost(cost_bps),
        initial_equity=INITIAL_EQUITY,
        periods_per_year=PERIODS_PER_YEAR,
    )
    return result


def _fmt(v: float) -> str:
    if v != v:
        return "n/a"
    return f"{v:+,.3f}" if abs(v) < 10 else f"{v:+,.1f}"


def main() -> None:
    for label, start, end in [
        ("HISTORICAL 2020-2025", "2020-01-01", "2025-01-01"),
        ("RECENT 2025-2026",     "2025-01-01", "2026-09-23"),
    ]:
        print(f"\n{'=' * 88}")
        print(f"  BTC/ETH Pair Spread Mean Reversion — {label}")
        print(f"{'=' * 88}")

        try:
            prices = _load_pair(start, end)
        except FileNotFoundError as e:
            print(f"  Skipped: {e}")
            continue

        days = (prices.index[-1] - prices.index[0]).days
        years = days / 365.25
        print(f"  Bars: {len(prices)}  ~{years:.2f} years  |  Cost: {COST_BPS:.1f} bps/side")

        print(f"\n  {'lookback':>9s}  {'entry_z':>8s}  {'exit_z':>7s}  "
              f"{'Final$':>11s}  {'TotRet':>9s}  {'CAGR':>8s}  {'Sharpe':>8s}  {'MaxDD':>9s}")

        for params in GRID:
            try:
                r = _run(prices, params)
                m = r.metrics
                print(
                    f"  {params['lookback']:>9d}  {params['entry_z']:>8.1f}  "
                    f"{params['exit_z']:>7.1f}  "
                    f"{r.equity.iloc[-1]:>11,.0f}  "
                    f"{m['total_return']:>+8.1%}  "
                    f"{m['cagr']:>+7.1%}  {m['sharpe']:>+8.2f}  "
                    f"{m['max_drawdown']:>+9.2%}"
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Params {} failed: {}", params, exc)

        # Also test cost sensitivity on the base config (42/2.0/0.5).
        print(f"\n  Cost sensitivity (lookback=42, entry_z=2.0, exit_z=0.5):")
        print(f"  {'bps/side':>9s}  {'Sharpe':>8s}  {'TotRet':>9s}  {'MaxDD':>9s}")
        for bps in [5.0, 10.0, 15.0, 25.0]:
            r = _run(prices, {"lookback": 42, "entry_z": 2.0, "exit_z": 0.5}, cost_bps=bps)
            m = r.metrics
            print(
                f"  {bps:>9.1f}  {m['sharpe']:>+8.2f}  "
                f"{m['total_return']:>+8.1%}  {m['max_drawdown']:>+9.2%}"
            )


if __name__ == "__main__":
    main()
    
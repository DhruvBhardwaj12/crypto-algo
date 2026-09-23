"""Backtest filtered cross-sectional momentum — historical + recent windows.

Run:
    uv run python scripts/backtest_cs_filtered.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml
from loguru import logger

from crypto_algo.backtesting.portfolio_simulator import run_portfolio_backtest
from crypto_algo.data.loaders import load_parquet
from crypto_algo.strategies.cs_momentum_filtered import build_filtered_weights

INITIAL_EQUITY = 10_000.0
PERIODS_PER_YEAR = 6 * 365
COST_BPS = 5.0


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


def _run(prices: pd.DataFrame, cost_bps: float = COST_BPS):
    weights = build_filtered_weights(prices)
    return run_portfolio_backtest(
        prices=prices,
        weights=weights,
        cost_model=_Cost(cost_bps),
        initial_equity=INITIAL_EQUITY,
        periods_per_year=PERIODS_PER_YEAR,
    )


def _fmt(v: float) -> str:
    if v != v:
        return "n/a"
    return f"{v:+,.3f}" if abs(v) < 10 else f"{v:+,.2f}"


def main() -> None:
    with open("config/settings.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)["data"]

    for label, start, end in [
        ("HISTORICAL 2020-2025", "2020-01-01", "2025-01-01"),
        ("RECENT 2025-2026",     "2025-01-01", "2026-09-23"),
    ]:
        print(f"\n{'=' * 82}")
        print(f"  Filtered CS Momentum (long-only, top-3, daily) — {label}")
        print(f"{'=' * 82}")

        try:
            prices = _load_all(cfg["symbols"], start, end)
        except ValueError as e:
            print(f"  Skipped: {e}")
            continue

        days = (prices.index[-1] - prices.index[0]).days
        years = days / 365.25
        print(f"  Universe: {prices.shape[1]} symbols | Bars: {prices.shape[0]} | ~{years:.2f} years")

        # Note: signals need 200 days of history for MA_200 to form. For the
        # 2020-2025 window we have enough. For 2025-2026 alone, we don't.
        # In production, we'd fetch pre-2025 data for the warmup. For now,
        # we accept that the recent window will show fewer trading days
        # early on until MA_200 has enough data.

        result = _run(prices)
        m = result.metrics

        # Compute per-symbol exposure summary
        w = result.weights
        n_positions_avg = (w > 0).sum(axis=1).mean()
        pct_time_flat = float((w.sum(axis=1) == 0).mean())
        avg_exposure = float(w.sum(axis=1).mean())

        print(f"\n  Final equity:      {result.equity.iloc[-1]:>12,.2f}  (start {INITIAL_EQUITY:,.2f})")
        print(f"  Total return:      {m['total_return']:>+10.1%}")
        print(f"  CAGR:              {m['cagr']:>+10.1%}")
        print(f"  Sharpe:            {m['sharpe']:>+10.2f}")
        print(f"  Sortino:           {m['sortino']:>+10.2f}")
        print(f"  Max drawdown:      {m['max_drawdown']:>+10.2%}")
        print(f"  Calmar:            {m['calmar']:>+10.2f}")
        print(f"  Avg positions/day: {n_positions_avg:>10.2f}")
        print(f"  Avg exposure:      {avg_exposure:>10.2f}  (1.0 = fully long)")
        print(f"  % time flat:       {pct_time_flat:>10.1%}")

        # Cost sensitivity
        print(f"\n  Cost sensitivity:")
        print(f"  {'bps/side':>9s}  {'Sharpe':>8s}  {'TotRet':>10s}  {'MaxDD':>9s}")
        for bps in [5.0, 10.0, 15.0, 25.0]:
            r = _run(prices, cost_bps=bps)
            mm = r.metrics
            print(f"  {bps:>9.1f}  {mm['sharpe']:>+8.2f}  "
                  f"{mm['total_return']:>+9.1%}  {mm['max_drawdown']:>+9.2%}")


if __name__ == "__main__":
    main()
    
"""Backtest Leveraged Trend Rider across multiple symbols and leverage levels.

Run:
    uv run python scripts/backtest_leveraged_trend.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from crypto_algo.backtesting.leveraged_simulator import run_leveraged_backtest
from crypto_algo.data.loaders import load_parquet
from crypto_algo.strategies.leveraged_trend import leveraged_trend_signal

INITIAL_EQUITY = 100.0  # ~₹8,300 per symbol (if we had 20k across 10 symbols, that's 2k each)
PERIODS_PER_YEAR = 6 * 365
COST_BPS = 5.0
LEVERAGE_LEVELS = [1.0, 2.0, 3.0, 5.0]

# Liquid majors
SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
    "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "LINKUSDT", "DOTUSDT",
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


def _load(symbol: str, start: str, end: str) -> pd.DataFrame | None:
    path = Path(f"data/raw/binance/futures/{symbol}/4h/{start}_{end}.parquet")
    if not path.exists():
        return None
    return load_parquet(path)


def _fmt(v: float, pct: bool = False) -> str:
    if v != v:
        return "n/a"
    if pct:
        return f"{v:+.1%}"
    return f"{v:+.2f}" if abs(v) < 100 else f"{v:+,.0f}"


def run_window(start: str, end: str, label: str) -> None:
    print(f"\n{'=' * 100}")
    print(f"  Leveraged Trend Rider — {label} ({start} to {end})")
    print(f"{'=' * 100}")

    # Load data for all symbols
    loaded = {}
    for s in SYMBOLS:
        df = _load(s, start, end)
        if df is not None and len(df) > 200:
            loaded[s] = df

    if not loaded:
        print("  No symbols loaded.")
        return

    print(f"  Symbols: {len(loaded)} | Initial equity per symbol: ${INITIAL_EQUITY:.2f}")

    for leverage in LEVERAGE_LEVELS:
        print(f"\n  --- Leverage {leverage:.1f}x ---")
        print(f"  {'symbol':>10s}  {'bars':>6s}  {'trades':>7s}  {'liq':>4s}  "
              f"{'total_ret':>10s}  {'sharpe':>7s}  {'max_dd':>8s}  "
              f"{'avg_month':>10s}  {'pos_months':>11s}")

        per_symbol_equity = []
        total_trades = 0
        total_liquidations = 0

        for symbol, df in loaded.items():
            signal = leveraged_trend_signal(df)
            result = run_leveraged_backtest(
                df=df,
                target_position=signal,
                leverage=leverage,
                cost_model=_Cost(COST_BPS),
                initial_equity=INITIAL_EQUITY,
                periods_per_year=PERIODS_PER_YEAR,
            )
            m = result.metrics
            total_trades += result.n_trades
            total_liquidations += result.n_liquidations

            print(
                f"  {symbol:>10s}  {len(df):>6d}  {result.n_trades:>7d}  "
                f"{result.n_liquidations:>4d}  "
                f"{m['total_return']:>+9.1%}  {m['sharpe']:>+7.2f}  "
                f"{m['max_drawdown']:>+7.1%}  "
                f"{m['avg_monthly_return']:>+9.2%}  "
                f"{m['pct_positive_months']:>10.0%}"
            )

            per_symbol_equity.append(result.equity)

        # Equal-weight portfolio: average normalized equity curves
        min_len = min(len(e) for e in per_symbol_equity)
        normalized = [e.iloc[:min_len] / e.iloc[0] for e in per_symbol_equity]
        combined = pd.concat(normalized, axis=1).mean(axis=1) * INITIAL_EQUITY

        rets = combined.pct_change().dropna()
        total = float(combined.iloc[-1] / combined.iloc[0] - 1.0)
        std = float(rets.std())
        sharpe = float(rets.mean() / std * np.sqrt(PERIODS_PER_YEAR)) if std > 0 else float("nan")
        peak = combined.cummax()
        max_dd = float((combined / peak - 1.0).min())
        avg_monthly = float(rets.resample("30D").apply(lambda r: (1 + r).prod() - 1).mean())

        print(f"\n  PORTFOLIO (equal-weight {len(loaded)} symbols):")
        print(f"    total return:    {total:>+10.1%}")
        print(f"    Sharpe:          {sharpe:>+10.2f}")
        print(f"    max drawdown:    {max_dd:>+10.2%}")
        print(f"    avg monthly:     {avg_monthly:>+10.2%}")
        print(f"    total trades:    {total_trades}")
        print(f"    liquidations:    {total_liquidations}")


def main() -> None:
    run_window("2020-01-01", "2025-01-01", "HISTORICAL")
    run_window("2025-01-01", "2026-09-23", "RECENT")


if __name__ == "__main__":
    main()
    
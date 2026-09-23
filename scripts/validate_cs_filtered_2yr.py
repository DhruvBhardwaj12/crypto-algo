"""Filtered CS Momentum — extended "modern era" test.

Concatenates 2024 (from the historical file) with 2025-2026 (from the
recent file) to give a 2.7-year window representing the ETF/institutional
era. Compares against buy-and-hold BTC.

Run:
    uv run python scripts/validate_cs_filtered_2yr.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from loguru import logger

from crypto_algo.backtesting.portfolio_simulator import run_portfolio_backtest
from crypto_algo.data.loaders import load_parquet
from crypto_algo.strategies.cs_momentum_filtered import build_filtered_weights
from crypto_algo.validation.bootstrap import block_bootstrap_sharpe

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


def _load_extended(symbols: list[str]) -> pd.DataFrame:
    """Concatenate 2024 from hist file with 2025-2026 from recent file."""
    series = {}
    for symbol in symbols:
        # Historical piece (use only 2024-01 onwards for the modern era)
        hist_path = Path(f"data/raw/binance/futures/{symbol}/4h/2020-01-01_2025-01-01.parquet")
        recent_path = Path(f"data/raw/binance/futures/{symbol}/4h/2025-01-01_2026-09-23.parquet")
        if not hist_path.exists() or not recent_path.exists():
            continue

        df_hist = load_parquet(hist_path).set_index("open_time")
        df_recent = load_parquet(recent_path).set_index("open_time")
        df_hist = df_hist.loc[df_hist.index >= pd.Timestamp("2024-01-01", tz="UTC")]

        combined = pd.concat([df_hist["close"], df_recent["close"]])
        combined = combined[~combined.index.duplicated(keep="first")].sort_index()
        if len(combined) > 100:
            series[symbol] = combined.rename(symbol)

    if not series:
        raise ValueError("No data available for extended window.")
    wide = pd.DataFrame(series).sort_index()
    non_nan = wide.notna().sum(axis=1)
    valid_start = non_nan[non_nan >= 6].index.min()
    return wide.loc[wide.index >= valid_start]


def _run(prices: pd.DataFrame, cost_bps: float = COST_BPS):
    weights = build_filtered_weights(prices)
    return run_portfolio_backtest(
        prices=prices, weights=weights, cost_model=_Cost(cost_bps),
        initial_equity=INITIAL_EQUITY, periods_per_year=PERIODS_PER_YEAR,
    )


def _slice_stats(equity: pd.Series) -> dict:
    if len(equity) < 10:
        return {"ret": float("nan"), "sharpe": float("nan"), "max_dd": float("nan")}
    rets = equity.pct_change().dropna()
    total = float(equity.iloc[-1] / equity.iloc[0] - 1.0)
    std = float(rets.std())
    sharpe = float(rets.mean() / std * np.sqrt(PERIODS_PER_YEAR)) if std > 0 else float("nan")
    peak = equity.cummax()
    dd = float((equity / peak - 1.0).min())
    return {"ret": total, "sharpe": sharpe, "max_dd": dd}


def main() -> None:
    with open("config/settings.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)["data"]

    print(f"\n{'=' * 82}")
    print(f"  Filtered CS Momentum — Extended modern-era test")
    print(f"{'=' * 82}")

    prices = _load_extended(cfg["symbols"])
    days = (prices.index[-1] - prices.index[0]).days
    years = days / 365.25

    print(f"  Universe: {prices.shape[1]} symbols | Bars: {prices.shape[0]} | ~{years:.2f} years")
    print(f"  Window: {prices.index[0].date()} to {prices.index[-1].date()}")

    # Strategy
    result = _run(prices)
    m = result.metrics

    # Buy & hold BTC for comparison
    btc = prices["BTCUSDT"].dropna()
    btc_ret = float(btc.iloc[-1] / btc.iloc[0] - 1.0)
    btc_rets = btc.pct_change().dropna()
    btc_std = float(btc_rets.std())
    btc_sharpe = float(btc_rets.mean() / btc_std * np.sqrt(PERIODS_PER_YEAR)) if btc_std > 0 else float("nan")
    btc_peak = btc.cummax()
    btc_dd = float((btc / btc_peak - 1.0).min())

    print(f"\n  {'':20s}  {'Strategy':>12s}  {'BTC B&H':>12s}")
    print(f"  {'Total return':20s}  {m['total_return']:>+11.1%}  {btc_ret:>+11.1%}")
    print(f"  {'CAGR':20s}  {m['cagr']:>+11.1%}  "
          f"{((btc.iloc[-1]/btc.iloc[0]) ** (365.25/days) - 1):>+11.1%}")
    print(f"  {'Sharpe':20s}  {m['sharpe']:>+11.2f}  {btc_sharpe:>+11.2f}")
    print(f"  {'Max drawdown':20s}  {m['max_drawdown']:>+11.2%}  {btc_dd:>+11.2%}")
    print(f"  {'Calmar':20s}  {m['calmar']:>+11.2f}  "
          f"{(m['cagr'] / abs(btc_dd) if btc_dd < 0 else float('nan')):>+11.2f}")

    # Rolling windows
    print(f"\n  Rolling 5-window breakdown:")
    eq = result.equity
    n = len(eq)
    bounds = np.linspace(0, n, 6, dtype=int)
    n_pos = 0
    for i in range(5):
        s = eq.iloc[bounds[i]:bounds[i + 1]]
        if len(s) < 50:
            continue
        stats = _slice_stats(s)
        if stats["ret"] > 0:
            n_pos += 1
        print(f"    {str(s.index[0].date()):>12s} - {str(s.index[-1].date()):>10s}  "
              f"ret={stats['ret']:>+7.1%}  sharpe={stats['sharpe']:>+6.2f}  "
              f"maxdd={stats['max_dd']:>+7.2%}")
    print(f"  windows profitable: {n_pos}/5")

    # Quarterly
    print(f"\n  Quarterly breakdown:")
    rets_q = eq.pct_change().fillna(0.0)
    rets_q.index = pd.to_datetime(rets_q.index, utc=True)
    n_pos_q = 0
    n_total_q = 0
    for q, s in rets_q.groupby(pd.Grouper(freq="QE")):
        if len(s) < 50:
            continue
        total = float((1 + s).prod() - 1)
        std = float(s.std())
        sh = float(s.mean() / std * np.sqrt(PERIODS_PER_YEAR)) if std > 0 else float("nan")
        n_total_q += 1
        if total > 0:
            n_pos_q += 1
        print(f"    {str(q.date()):>13s}  {total:>+8.2%}  sharpe={sh:>+6.2f}  bars={len(s)}")
    print(f"  quarters profitable: {n_pos_q}/{n_total_q}")

    # Bootstrap
    print(f"\n  Bootstrap (2000 resamples, block=30):")
    bs = block_bootstrap_sharpe(result.returns, periods_per_year=PERIODS_PER_YEAR)
    print(f"    Sharpe point:    {bs['sharpe_point']:+.3f}")
    print(f"    5th percentile:  {bs['p05']:+.3f}")
    print(f"    50th percentile: {bs['p50']:+.3f}")
    print(f"    95th percentile: {bs['p95']:+.3f}")
    print(f"    P(Sharpe > 0):   {bs['pct_above_zero']:.1%}")

    # Cost sensitivity
    print(f"\n  Cost sensitivity:")
    for bps in [5.0, 10.0, 15.0, 25.0]:
        r = _run(prices, cost_bps=bps)
        mm = r.metrics
        print(f"    {bps:>5.1f} bps/side  Sharpe={mm['sharpe']:>+6.2f}  "
              f"ret={mm['total_return']:>+8.1%}  maxdd={mm['max_drawdown']:>+7.2%}")

    # Exposure
    w = result.weights
    total_exposure = w.sum(axis=1)
    print(f"\n  Exposure:")
    print(f"    Avg gross exposure:  {total_exposure.mean():.3f}")
    print(f"    % bars flat:         {float((total_exposure < 1e-6).mean()):.1%}")


if __name__ == "__main__":
    main()
    
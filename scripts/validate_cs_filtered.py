"""Validation for the filtered cross-sectional momentum strategy.

All parameters were fixed in advance (MA periods, lookbacks, top-N).
No tuning. So instead of a parameter-search walk-forward, we test
CONSISTENCY: split the period into windows, check per-window results.

Tests:
  [1] Window breakdown (5 windows on historical, 4 on recent)
  [2] Quarterly breakdown (recent window, finer granularity)
  [3] Bootstrap on recent returns
  [4] Cost sensitivity (already validated, but re-confirm)
  [5] Regime state analysis: what fraction of time is the strategy active?

Run:
    uv run python scripts/validate_cs_filtered.py
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


def _slice_stats(equity: pd.Series) -> dict:
    """Given an equity slice, return return/sharpe/max_dd."""
    if len(equity) < 10:
        return {"ret": float("nan"), "sharpe": float("nan"), "max_dd": float("nan")}
    rets = equity.pct_change().dropna()
    total = float(equity.iloc[-1] / equity.iloc[0] - 1.0)
    std = float(rets.std())
    sharpe = float(rets.mean() / std * np.sqrt(PERIODS_PER_YEAR)) if std > 0 else float("nan")
    peak = equity.cummax()
    dd = float((equity / peak - 1.0).min())
    return {"ret": total, "sharpe": sharpe, "max_dd": dd}


def window_breakdown(prices: pd.DataFrame, label: str, n_windows: int) -> None:
    result = _run(prices)
    eq = result.equity

    print(f"\n  [{label}] Rolling {n_windows}-window breakdown:")
    print(f"  {'window':>25s}  {'return':>9s}  {'sharpe':>8s}  {'max_dd':>9s}  {'bars':>6s}")

    # Use quarter-like boundaries.
    n = len(eq)
    bounds = np.linspace(0, n, n_windows + 1, dtype=int)
    n_pos = 0
    for i in range(n_windows):
        s = eq.iloc[bounds[i]:bounds[i + 1]]
        if len(s) < 50:
            continue
        stats = _slice_stats(s)
        if stats["ret"] > 0:
            n_pos += 1
        print(
            f"  {str(s.index[0].date()):>12s} - {str(s.index[-1].date()):>10s}  "
            f"{stats['ret']:>+8.1%}  {stats['sharpe']:>+8.2f}  "
            f"{stats['max_dd']:>+9.2%}  {len(s):>6d}"
        )

    print(f"  windows profitable: {n_pos}/{n_windows}")


def quarterly_breakdown(prices: pd.DataFrame, label: str) -> None:
    result = _run(prices)
    eq = result.equity
    rets = eq.pct_change().fillna(0.0)
    rets.index = pd.to_datetime(rets.index, utc=True)

    print(f"\n  [{label}] Quarterly breakdown:")
    print(f"  {'quarter_end':>13s}  {'return':>9s}  {'sharpe':>8s}  {'bars':>6s}")
    n_pos = 0
    n_total = 0
    grouped = rets.groupby(pd.Grouper(freq="QE"))
    for q, s in grouped:
        if len(s) < 50:
            continue
        total = float((1 + s).prod() - 1)
        std = float(s.std())
        sh = float(s.mean() / std * np.sqrt(PERIODS_PER_YEAR)) if std > 0 else float("nan")
        n_total += 1
        if total > 0:
            n_pos += 1
        print(f"  {str(q.date()):>13s}  {total:>+8.2%}  {sh:>+8.2f}  {len(s):>6d}")
    print(f"  quarters profitable: {n_pos}/{n_total}")


def exposure_analysis(prices: pd.DataFrame, label: str) -> None:
    result = _run(prices)
    w = result.weights
    total_exposure = w.sum(axis=1)

    print(f"\n  [{label}] Exposure analysis:")
    print(f"    Avg gross exposure:      {total_exposure.mean():.3f}")
    print(f"    Median gross exposure:   {total_exposure.median():.3f}")
    print(f"    % of bars flat:          {float((total_exposure < 1e-6).mean()):.1%}")
    print(f"    % of bars with 1 pos:    {float(((w > 0).sum(axis=1) == 1).mean()):.1%}")
    print(f"    % of bars with 2 pos:    {float(((w > 0).sum(axis=1) == 2).mean()):.1%}")
    print(f"    % of bars with 3 pos:    {float(((w > 0).sum(axis=1) == 3).mean()):.1%}")

    # Top symbols traded by time in position
    in_pos_by_sym = (w > 0).sum() / len(w)
    top_syms = in_pos_by_sym.sort_values(ascending=False).head(5)
    print(f"    Top 5 most-held symbols:")
    for sym, frac in top_syms.items():
        print(f"      {sym:>12s}  {frac:.1%} of bars")


def bootstrap_analysis(prices: pd.DataFrame, label: str) -> None:
    result = _run(prices)
    bs = block_bootstrap_sharpe(result.returns, periods_per_year=PERIODS_PER_YEAR)
    print(f"\n  [{label}] Bootstrap (2000 resamples, block=30):")
    print(f"    Sharpe point:    {bs['sharpe_point']:+.3f}")
    print(f"    5th percentile:  {bs['p05']:+.3f}")
    print(f"    50th percentile: {bs['p50']:+.3f}")
    print(f"    95th percentile: {bs['p95']:+.3f}")
    print(f"    P(Sharpe > 0):   {bs['pct_above_zero']:.1%}")


def main() -> None:
    with open("config/settings.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)["data"]

    windows = [
        ("HISTORICAL", "2020-01-01", "2025-01-01"),
        ("RECENT",     "2025-01-01", "2026-09-23"),
    ]

    for label, start, end in windows:
        print(f"\n{'=' * 82}")
        print(f"  Filtered CS Momentum — Validation — {label} ({start} to {end})")
        print(f"{'=' * 82}")

        try:
            prices = _load_all(cfg["symbols"], start, end)
        except ValueError as e:
            print(f"  Skipped: {e}")
            continue

        days = (prices.index[-1] - prices.index[0]).days
        years = days / 365.25
        print(f"  Universe: {prices.shape[1]} symbols | Bars: {prices.shape[0]} | ~{years:.2f} years")

        # For the recent window, MA_200 takes 200 days to form. So the
        # effective trading window is shorter. Note that in the output.

        # [1] Window breakdown
        n_windows = 5 if label == "HISTORICAL" else 4
        window_breakdown(prices, label, n_windows)

        # [2] Quarterly
        quarterly_breakdown(prices, label)


        # [3] Bootstrap
        bootstrap_analysis(prices, label)

        # [4] Exposure
        exposure_analysis(prices, label)


if __name__ == "__main__":
    main()
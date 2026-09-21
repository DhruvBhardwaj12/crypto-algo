"""Regime and calendar breakdown.

A strategy that earns all its money in one regime (a single bull market)
is fragile, not robust. We split performance two ways:

1. Calendar year — trivially interpretable.
2. Volatility regime — days bucketed into low/mid/high volatility terciles.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from crypto_algo.backtesting.simulator import BacktestResult


def _annualized(returns: pd.Series, periods_per_year: int) -> dict:
    if returns.empty:
        return {"total_return": float("nan"), "sharpe": float("nan"), "n_days": 0}
    total = float((1 + returns).prod() - 1)
    std = float(returns.std())
    sharpe = (
        float(returns.mean() / std * np.sqrt(periods_per_year)) if std > 0 else float("nan")
    )
    return {"total_return": total, "sharpe": sharpe, "n_days": len(returns)}


def calendar_year_breakdown(
    result: BacktestResult,
    benchmark_returns: pd.Series,
    periods_per_year: int,
) -> pd.DataFrame:
    """Per-year strategy vs benchmark."""
    strat = result.returns.copy()
    bench = benchmark_returns.copy()
    strat.index = pd.to_datetime(strat.index, utc=True)
    bench.index = pd.to_datetime(bench.index, utc=True)

    years = sorted(set(strat.index.year))
    rows = []
    for y in years:
        s = strat[strat.index.year == y]
        b = bench[bench.index.year == y]
        s_stats = _annualized(s, periods_per_year)
        b_stats = _annualized(b, periods_per_year)
        rows.append({
            "year": y,
            "strat_return": s_stats["total_return"],
            "strat_sharpe": s_stats["sharpe"],
            "bench_return": b_stats["total_return"],
            "bench_sharpe": b_stats["sharpe"],
            "n_days": s_stats["n_days"],
        })
    return pd.DataFrame(rows)


def volatility_regime_breakdown(
    result: BacktestResult,
    underlying_returns: pd.Series,
    periods_per_year: int,
    window: int = 30,
) -> pd.DataFrame:
    """Bucket bars by rolling volatility tercile of the underlying."""
    vol = underlying_returns.rolling(window, min_periods=window).std().dropna()
    if vol.empty:
        return pd.DataFrame()

    q1, q2 = vol.quantile([1 / 3, 2 / 3]).tolist()

    def bucket(v: float) -> str:
        if v <= q1:
            return "low_vol"
        if v <= q2:
            return "mid_vol"
        return "high_vol"

    regime = vol.apply(bucket)

    strat = result.returns.copy()
    strat.index = pd.to_datetime(strat.index, utc=True)

    rows = []
    for name in ["low_vol", "mid_vol", "high_vol"]:
        idx = regime[regime == name].index
        s = strat.reindex(idx).dropna()
        stats = _annualized(s, periods_per_year)
        rows.append({
            "regime": name,
            "n_days": stats["n_days"],
            "return": stats["total_return"],
            "sharpe": stats["sharpe"],
        })
    return pd.DataFrame(rows)


def format_regime_output(
    cal: pd.DataFrame,
    vol: pd.DataFrame,
) -> str:
    lines = ["  Calendar years:", "    year   strat      bench      strat_sharpe  bench_sharpe"]
    for _, r in cal.iterrows():
        lines.append(
            f"    {int(r['year'])}   {r['strat_return']:+.1%}   "
            f"{r['bench_return']:+.1%}    {r['strat_sharpe']:+.2f}       {r['bench_sharpe']:+.2f}"
        )
    if not vol.empty:
        lines.append("  Volatility regimes (of underlying):")
        for _, r in vol.iterrows():
            lines.append(
                f"    {r['regime']:9s}  days={int(r['n_days']):4d}  "
                f"ret={r['return']:+.2%}  sharpe={r['sharpe']:+.2f}"
            )
    return "\n".join(lines)

    
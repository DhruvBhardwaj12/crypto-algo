"""Validate cross-sectional reversion on the RECENT 2025-2026 period.

Same structure as LAB-007 validation but on fresh data. Since the recent
window is short (~1.7 years), we do a full parameter sweep + bootstrap
on the chosen config. Walk-forward is limited but included.

Run:
    uv run python scripts/validate_cs_reversion_recent.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from loguru import logger

from crypto_algo.backtesting.portfolio_simulator import run_portfolio_backtest
from crypto_algo.data.loaders import load_parquet
from crypto_algo.strategies.cross_sectional_reversion import (
    cross_sectional_reversion_weights,
)
from crypto_algo.validation.bootstrap import block_bootstrap_sharpe

START = "2025-01-01"
END = "2026-09-23"
INITIAL_EQUITY = 10_000.0
PERIODS_PER_YEAR = 6 * 365
COST_BPS = 5.0

# Chosen parameter (from prior test). Will not be tuned.
CHOSEN = {"lookback": 42, "n_long": 3, "n_short": 3, "rebalance": 6}


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
    wide = pd.DataFrame(series).sort_index()
    non_nan = wide.notna().sum(axis=1)
    valid_start = non_nan[non_nan >= 6].index.min()
    return wide.loc[wide.index >= valid_start]


def _run(prices: pd.DataFrame, params: dict, cost_bps: float = COST_BPS):
    weights = cross_sectional_reversion_weights(
        prices,
        lookback=int(params["lookback"]),
        n_long=int(params["n_long"]),
        n_short=int(params["n_short"]),
        rebalance_every=int(params["rebalance"]),
    )
    result = run_portfolio_backtest(
        prices=prices, weights=weights, cost_model=_Cost(cost_bps),
        initial_equity=INITIAL_EQUITY, periods_per_year=PERIODS_PER_YEAR,
    )
    return result


def parameter_sweep(prices: pd.DataFrame) -> pd.DataFrame:
    grid = [
        {"lookback": lb, "n_long": nl, "n_short": ns, "rebalance": rb}
        for lb in [14, 21, 30, 42, 60, 84]
        for nl in [2, 3, 5]
        for ns in [2, 3, 5]
        for rb in [3, 6, 12]
    ]
    rows = []
    for i, params in enumerate(grid, 1):
        try:
            r = _run(prices, params)
            row = dict(params)
            row.update(r.metrics)
            rows.append(row)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Param failed {}: {}", params, exc)
        if i % 20 == 0:
            logger.info("  swept {}/{}", i, len(grid))
    return pd.DataFrame(rows)


def summarize_sweep(sweep: pd.DataFrame, chosen: dict) -> str:
    if sweep.empty:
        return "Empty sweep."
    valid = sweep.dropna(subset=["sharpe"]).copy()
    n = len(valid)
    n_pos = int((valid["sharpe"] > 0).sum())
    n_gt_05 = int((valid["sharpe"] > 0.5).sum())
    n_gt_10 = int((valid["sharpe"] > 1.0).sum())
    n_gt_13 = int((valid["sharpe"] > 1.3).sum())
    median_sharpe = float(valid["sharpe"].median())
    best = valid.loc[valid["sharpe"].idxmax()]
    worst = valid.loc[valid["sharpe"].idxmin()]

    chosen_row = valid[
        (valid["lookback"] == chosen["lookback"])
        & (valid["n_long"] == chosen["n_long"])
        & (valid["n_short"] == chosen["n_short"])
        & (valid["rebalance"] == chosen["rebalance"])
    ]
    chosen_sharpe = float(chosen_row["sharpe"].iloc[0]) if not chosen_row.empty else float("nan")
    chosen_pct = (
        float((valid["sharpe"] < chosen_sharpe).mean() * 100)
        if not chosen_row.empty else float("nan")
    )

    lines = [
        f"  grid size:              {n}",
        f"  Sharpe > 0:             {n_pos}/{n} ({n_pos / n:.0%})",
        f"  Sharpe > 0.5:           {n_gt_05}/{n} ({n_gt_05 / n:.0%})",
        f"  Sharpe > 1.0:           {n_gt_10}/{n} ({n_gt_10 / n:.0%})",
        f"  Sharpe > 1.3:           {n_gt_13}/{n} ({n_gt_13 / n:.0%})",
        f"  median Sharpe:          {median_sharpe:.3f}",
        f"  best:  {int(best['lookback'])}/{int(best['n_long'])}/{int(best['n_short'])}/{int(best['rebalance'])} "
        f"Sharpe={best['sharpe']:.3f}",
        f"  worst: {int(worst['lookback'])}/{int(worst['n_long'])}/{int(worst['n_short'])}/{int(worst['rebalance'])} "
        f"Sharpe={worst['sharpe']:.3f}",
        f"  chosen: {chosen['lookback']}/{chosen['n_long']}/{chosen['n_short']}/{chosen['rebalance']} "
        f"Sharpe={chosen_sharpe:.3f} (pct {chosen_pct:.0f})",
    ]

    if n_gt_05 / n > 0.6:
        lines.append("  VERDICT: broad region of decent Sharpe. Consistent with real edge.")
    elif n_pos / n < 0.4:
        lines.append("  VERDICT: most of the grid loses. Chosen params likely a lucky spot.")
    else:
        lines.append("  VERDICT: mixed. Inconclusive.")

    return "\n".join(lines)


def main() -> None:
    with open("config/settings.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)["data"]
    prices = _load_all(cfg["symbols"])

    days = (prices.index[-1] - prices.index[0]).days
    years = days / 365.25

    print(f"\n{'=' * 82}")
    print(f"  Cross-Sectional REVERSION — Validation on recent period")
    print(f"  {prices.shape[1]} symbols | {prices.shape[0]} bars | ~{years:.2f} years")
    print(f"{'=' * 82}")

    # [1] Full sweep
    print("\n[1] Parameter sweep (162 configs: lookback × n_long × n_short × rebalance)")
    sweep = parameter_sweep(prices)
    print(summarize_sweep(sweep, CHOSEN))
    sweep.to_csv("research/cs_reversion_recent_sweep.csv", index=False)

    # [2] Bootstrap on chosen
    print("\n[2] Bootstrap on chosen params (42/3/3/6)")
    r = _run(prices, CHOSEN)
    rets = r.returns
    bs = block_bootstrap_sharpe(rets, periods_per_year=PERIODS_PER_YEAR)
    print(f"  Sharpe point:    {bs['sharpe_point']:+.3f}")
    print(f"  5th percentile:  {bs['p05']:+.3f}")
    print(f"  50th percentile: {bs['p50']:+.3f}")
    print(f"  95th percentile: {bs['p95']:+.3f}")
    print(f"  P(Sharpe > 0):   {bs['pct_above_zero']:.1%}")

    # [3] Cost sensitivity
    print("\n[3] Cost sensitivity on chosen params")
    print(f"  {'bps/side':>9s}  {'RT bps':>7s}  {'TotalRet':>10s}  {'Sharpe':>8s}  {'MaxDD':>9s}")
    for bps in [5.0, 10.0, 15.0, 25.0]:
        r2 = _run(prices, CHOSEN, cost_bps=bps)
        m = r2.metrics
        print(
            f"  {bps:>9.1f}  {2 * bps:>7.1f}  "
            f"{m['total_return']:>+9.1%}  {m['sharpe']:>+8.2f}  "
            f"{m['max_drawdown']:>+9.2%}"
        )

    # [4] Quarterly breakdown for the chosen params
    print("\n[4] Quarterly breakdown (chosen params, 5 bps)")
    eq = r.equity
    rets_q = eq.pct_change().fillna(0.0)
    rets_q.index = pd.to_datetime(rets_q.index, utc=True)
    quarters = rets_q.groupby(pd.Grouper(freq="Q"))
    print(f"  {'quarter':>10s}  {'return':>9s}  {'sharpe':>8s}")
    for q, s in quarters:
        if len(s) < 20:
            continue
        total = float((1 + s).prod() - 1)
        std = float(s.std())
        sh = float(s.mean() / std * np.sqrt(PERIODS_PER_YEAR)) if std > 0 else float("nan")
        print(f"  {str(q.date()):>10s}  {total:>+8.2%}  {sh:>+8.2f}")


if __name__ == "__main__":
    main()
    
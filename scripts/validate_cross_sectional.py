"""Full validation framework for LAB-007 (cross-sectional momentum).

Four tests:
  [1] Parameter sweep (lookback × n_long × n_short × rebalance)
  [2] Walk-forward (anchored, expanding)
  [3] Regime breakdown (calendar year + volatility)
  [4] Bootstrap on the return series

Run:
    uv run python scripts/validate_cross_sectional.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from loguru import logger

from crypto_algo.backtesting.portfolio_simulator import run_portfolio_backtest
from crypto_algo.data.loaders import load_parquet
from crypto_algo.strategies.cross_sectional_momentum import (
    cross_sectional_momentum_weights,
)
from crypto_algo.validation.bootstrap import block_bootstrap_sharpe

START = "2020-01-01"
END = "2025-01-01"
INITIAL_EQUITY = 10_000.0
PERIODS_PER_YEAR = 6 * 365
COST_BPS = 5.0

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


def _run(prices: pd.DataFrame, params: dict, cost_bps: float = COST_BPS) -> dict:
    weights = cross_sectional_momentum_weights(
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
    return {
        "final_equity": float(result.equity.iloc[-1]),
        "sharpe": float(result.metrics.get("sharpe", float("nan"))),
        "total_return": float(result.metrics.get("total_return", float("nan"))),
        "max_dd": float(result.metrics.get("max_drawdown", float("nan"))),
        "equity": result.equity,
        "returns": result.returns,
        "weights": result.weights,
    }


def _fmt(v: float) -> str:
    if v != v:
        return "n/a"
    if abs(v) < 10:
        return f"{v:+.4f}"
    return f"{v:+,.2f}"


def parameter_sweep(prices: pd.DataFrame) -> pd.DataFrame:
    grid = [
        {"lookback": lb, "n_long": nl, "n_short": ns, "rebalance": rb}
        for lb in [21, 42, 84]
        for nl in [2, 3, 5]
        for ns in [2, 3, 5]
        for rb in [3, 6, 12]
    ]
    rows = []
    for i, params in enumerate(grid, 1):
        try:
            r = _run(prices, params, cost_bps=COST_BPS)
            row = dict(params)
            row["sharpe"] = r["sharpe"]
            row["total_return"] = r["total_return"]
            row["max_dd"] = r["max_dd"]
            rows.append(row)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Param failed {}: {}", params, exc)
        if i % 10 == 0:
            logger.info("  sweep {}/{}", i, len(grid))
    return pd.DataFrame(rows)


def summarize_sweep(sweep: pd.DataFrame, chosen: dict) -> str:
    if sweep.empty:
        return "Empty sweep."
    valid = sweep.dropna(subset=["sharpe"])
    n = len(valid)
    n_pos = int((valid["sharpe"] > 0).sum())
    n_gt_05 = int((valid["sharpe"] > 0.5).sum())
    n_gt_10 = int((valid["sharpe"] > 1.0).sum())
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

    lines = [
        f"  grid size:              {n}",
        f"  Sharpe > 0:             {n_pos}/{n} ({n_pos / n:.0%})",
        f"  Sharpe > 0.5:           {n_gt_05}/{n} ({n_gt_05 / n:.0%})",
        f"  Sharpe > 1.0:           {n_gt_10}/{n} ({n_gt_10 / n:.0%})",
        f"  median Sharpe:          {median_sharpe:.3f}",
        f"  best:  {int(best['lookback'])}/{int(best['n_long'])}/{int(best['n_short'])}/{int(best['rebalance'])} "
        f"Sharpe={best['sharpe']:.3f}",
        f"  worst: {int(worst['lookback'])}/{int(worst['n_long'])}/{int(worst['n_short'])}/{int(worst['rebalance'])} "
        f"Sharpe={worst['sharpe']:.3f}",
        f"  chosen: {chosen['lookback']}/{chosen['n_long']}/{chosen['n_short']}/{chosen['rebalance']} "
        f"Sharpe={chosen_sharpe:.3f}",
    ]
    return "\n".join(lines)


def walk_forward(prices: pd.DataFrame, n_windows: int = 5) -> dict:
    start = prices.index.min()
    end = prices.index.max()
    boundaries = pd.date_range(start, end, periods=n_windows + 1, tz="UTC")

    grid = [
        {"lookback": lb, "n_long": nl, "n_short": ns, "rebalance": rb}
        for lb in [21, 42, 84]
        for nl in [2, 3, 5]
        for ns in [2, 3, 5]
        for rb in [3, 6, 12]
    ]

    compounded = 1.0
    n_profitable = 0
    n_total = 0
    test_sharpes = []

    for i in range(n_windows - 1):
        test_start = boundaries[i + 1]
        test_end = boundaries[i + 2] if i + 2 < len(boundaries) else end + pd.Timedelta("1s")

        train_prices = prices.loc[prices.index < test_start]
        test_prices = prices.loc[
            (prices.index >= test_start) & (prices.index < test_end)
        ]

        if len(train_prices) < 500 or len(test_prices) < 200:
            continue

        # Pick best params on train.
        train_rows = []
        for params in grid:
            try:
                r = _run(train_prices, params, cost_bps=COST_BPS)
                if not np.isnan(r["sharpe"]):
                    row = dict(params)
                    row["sharpe"] = r["sharpe"]
                    train_rows.append(row)
            except Exception:
                pass

        if not train_rows:
            continue
        train_df = pd.DataFrame(train_rows)
        best = train_df.loc[train_df["sharpe"].idxmax()]
        chosen = {
            "lookback": int(best["lookback"]),
            "n_long": int(best["n_long"]),
            "n_short": int(best["n_short"]),
            "rebalance": int(best["rebalance"]),
        }

        # Build combined (warmup from train tail) and run on test.
        warmup = train_prices.tail(200)
        combined = pd.concat([warmup, test_prices])
        r = _run(combined, chosen, cost_bps=COST_BPS)

        # Slice equity to test window.
        eq = r["equity"]
        eq = eq.loc[eq.index >= test_start]
        if len(eq) < 50:
            continue
        ret = float(eq.iloc[-1] / eq.iloc[0] - 1.0)
        rets = eq.pct_change().dropna()
        std = float(rets.std())
        sh = float(rets.mean() / std * np.sqrt(PERIODS_PER_YEAR)) if std > 0 else float("nan")

        compounded *= (1.0 + ret)
        n_total += 1
        if ret > 0:
            n_profitable += 1
        test_sharpes.append(sh)

        print(
            f"    {test_start.date()} -> {test_end.date()} "
            f"chosen={chosen['lookback']}/{chosen['n_long']}/{chosen['n_short']}/{chosen['rebalance']} "
            f"train Sh={float(best['sharpe']):+.2f} test Sh={sh:+.2f} ret={ret:+.1%}"
        )

    return {
        "compounded": compounded - 1.0,
        "mean_sharpe": float(np.mean(test_sharpes)) if test_sharpes else float("nan"),
        "windows_profitable": n_profitable / n_total if n_total > 0 else 0.0,
        "n_total": n_total,
    }


def regime_breakdown(prices: pd.DataFrame, params: dict) -> None:
    r = _run(prices, params, cost_bps=COST_BPS)
    eq = r["equity"]
    rets = eq.pct_change().fillna(0.0)
    rets.index = pd.to_datetime(rets.index, utc=True)

    print(f"  Calendar years:")
    print(f"    year    strat_ret    sharpe")
    for year in sorted(set(rets.index.year)):
        s = rets[rets.index.year == year]
        if len(s) < 20:
            continue
        total = float((1 + s).prod() - 1)
        std = float(s.std())
        sharpe = float(s.mean() / std * np.sqrt(PERIODS_PER_YEAR)) if std > 0 else float("nan")
        print(f"    {year}    {total:>+9.2%}    {sharpe:>+6.2f}")


def main() -> None:
    with open("config/settings.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)["data"]
    prices = _load_all(cfg["symbols"])

    print(f"\n{'=' * 78}")
    print(f"  Cross-Sectional Momentum — Full Validation")
    print(f"  {prices.shape[1]} symbols, {prices.shape[0]} bars")
    print(f"{'=' * 78}")

    # [1] Sweep
    print("\n[1] Parameter sweep (81 configs: lookback × n_long × n_short × rebalance)")
    sweep = parameter_sweep(prices)
    print(summarize_sweep(sweep, CHOSEN))
    sweep.to_csv("research/lab007_sweep.csv", index=False)

    # [2] Walk-forward
    print("\n[2] Walk-forward (anchored, expanding, 5 folds)")
    wf = walk_forward(prices, n_windows=5)
    print(f"  windows:               {wf['n_total']}")
    print(f"  compounded OOS return: {wf['compounded']:+.2%}")
    print(f"  mean test Sharpe:      {wf['mean_sharpe']:+.3f}")
    print(f"  windows profitable:    {wf['windows_profitable']:.0%}")

    # [3] Regime
    print("\n[3] Regime breakdown (chosen params)")
    regime_breakdown(prices, CHOSEN)

    # [4] Bootstrap
    print("\n[4] Bootstrap on chosen-params return series")
    r = _run(prices, CHOSEN, cost_bps=COST_BPS)
    bs = block_bootstrap_sharpe(r["returns"], periods_per_year=PERIODS_PER_YEAR)
    print(f"  Sharpe point:    {bs['sharpe_point']:+.3f}")
    print(f"  5th percentile:  {bs['p05']:+.3f}")
    print(f"  50th percentile: {bs['p50']:+.3f}")
    print(f"  95th percentile: {bs['p95']:+.3f}")
    print(f"  P(Sharpe > 0):   {bs['pct_above_zero']:.1%}")


if __name__ == "__main__":
    main()
    
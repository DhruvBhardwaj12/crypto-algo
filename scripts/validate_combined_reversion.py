"""Full validation for the combined reversion portfolio.

Combines CS Reversion + BTC/ETH Pair Spread (50/50). Tests:
  [1] Parameter sweep (both strategies' params jointly)
  [2] Walk-forward (rolling, anchored)
  [3] Regime / quarterly breakdown
  [4] Bootstrap
  [5] Cost sensitivity

Window: recent 2025-01-01 to 2026-09-23.

Run:
    uv run python scripts/validate_combined_reversion.py
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
from crypto_algo.strategies.pair_spread import pair_spread_weights
from crypto_algo.validation.bootstrap import block_bootstrap_sharpe

START = "2025-01-01"
END = "2026-09-23"
INITIAL_EQUITY = 10_000.0
PERIODS_PER_YEAR = 6 * 365
COST_BPS = 5.0

# Chosen from prior runs
CHOSEN = {
    "cs_lookback": 42, "cs_n_long": 3, "cs_n_short": 3, "cs_rebalance": 6,
    "pair_lookback": 42, "pair_entry_z": 2.0, "pair_exit_z": 0.5,
}


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
        raise ValueError(f"No data for {start} to {end}")
    wide = pd.DataFrame(series).sort_index()
    non_nan = wide.notna().sum(axis=1)
    valid_start = non_nan[non_nan >= 6].index.min()
    return wide.loc[wide.index >= valid_start]


def _build_weights(prices: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Build combined weights (50% CS Reversion + 50% Pair Spread)."""
    cs_w = cross_sectional_reversion_weights(
        prices,
        lookback=int(params["cs_lookback"]),
        n_long=int(params["cs_n_long"]),
        n_short=int(params["cs_n_short"]),
        rebalance_every=int(params["cs_rebalance"]),
    )

    pair_narrow = pair_spread_weights(
        prices["BTCUSDT"], prices["ETHUSDT"],
        lookback=int(params["pair_lookback"]),
        entry_z=float(params["pair_entry_z"]),
        exit_z=float(params["pair_exit_z"]),
        leg_weight=0.5,
    )
    pair_w = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    for col in ["BTCUSDT", "ETHUSDT"]:
        if col in pair_w.columns:
            pair_w[col] = pair_narrow[col].reindex(prices.index).fillna(0.0)

    return 0.5 * cs_w + 0.5 * pair_w


def _run(prices: pd.DataFrame, params: dict, cost_bps: float = COST_BPS):
    weights = _build_weights(prices, params)
    return run_portfolio_backtest(
        prices=prices, weights=weights, cost_model=_Cost(cost_bps),
        initial_equity=INITIAL_EQUITY, periods_per_year=PERIODS_PER_YEAR,
    )


def _metrics_from_equity(equity: pd.Series) -> dict:
    rets = equity.pct_change().dropna()
    total = float(equity.iloc[-1] / equity.iloc[0] - 1.0)
    years = len(equity) / PERIODS_PER_YEAR
    cagr = (
        float((equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1)
        if years > 0 and equity.iloc[-1] > 0 else float("nan")
    )
    std = float(rets.std())
    sharpe = float(rets.mean() / std * np.sqrt(PERIODS_PER_YEAR)) if std > 0 else float("nan")
    peak = equity.cummax()
    dd = float((equity / peak - 1.0).min())
    return {"total_return": total, "cagr": cagr, "sharpe": sharpe, "max_dd": dd}


def parameter_sweep(prices: pd.DataFrame) -> pd.DataFrame:
    grid = []
    for cs_lb in [21, 42, 84]:
        for cs_rb in [6, 12]:
            for pair_lb in [21, 42, 84]:
                for pair_entry in [1.5, 2.0]:
                    grid.append({
                        "cs_lookback": cs_lb, "cs_n_long": 3, "cs_n_short": 3,
                        "cs_rebalance": cs_rb,
                        "pair_lookback": pair_lb, "pair_entry_z": pair_entry,
                        "pair_exit_z": 0.5,
                    })
    rows = []
    for i, params in enumerate(grid, 1):
        try:
            r = _run(prices, params)
            row = dict(params)
            row["sharpe"] = float(r.metrics.get("sharpe", float("nan")))
            row["total_return"] = float(r.metrics.get("total_return", float("nan")))
            row["max_dd"] = float(r.metrics.get("max_drawdown", float("nan")))
            rows.append(row)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Param failed: {}", exc)
        if i % 12 == 0:
            logger.info("  swept {}/{}", i, len(grid))
    return pd.DataFrame(rows)


def _is_chosen(row: pd.Series) -> bool:
    return (
        row["cs_lookback"] == CHOSEN["cs_lookback"]
        and row["cs_rebalance"] == CHOSEN["cs_rebalance"]
        and row["pair_lookback"] == CHOSEN["pair_lookback"]
        and row["pair_entry_z"] == CHOSEN["pair_entry_z"]
    )


def summarize_sweep(sweep: pd.DataFrame) -> str:
    valid = sweep.dropna(subset=["sharpe"]).copy()
    n = len(valid)
    n_pos = int((valid["sharpe"] > 0).sum())
    n_gt_05 = int((valid["sharpe"] > 0.5).sum())
    n_gt_10 = int((valid["sharpe"] > 1.0).sum())
    median = float(valid["sharpe"].median())
    chosen = valid[valid.apply(_is_chosen, axis=1)]
    chosen_sharpe = float(chosen["sharpe"].iloc[0]) if not chosen.empty else float("nan")
    chosen_pct = float((valid["sharpe"] < chosen_sharpe).mean() * 100) if not chosen.empty else float("nan")

    lines = [
        f"  grid size:              {n}",
        f"  Sharpe > 0:             {n_pos}/{n} ({n_pos/n:.0%})",
        f"  Sharpe > 0.5:           {n_gt_05}/{n} ({n_gt_05/n:.0%})",
        f"  Sharpe > 1.0:           {n_gt_10}/{n} ({n_gt_10/n:.0%})",
        f"  median Sharpe:          {median:.3f}",
        f"  chosen Sharpe:          {chosen_sharpe:.3f} (pct {chosen_pct:.0f})",
    ]
    # Top 3 and bottom 3
    top = valid.nlargest(3, "sharpe")[["cs_lookback","cs_rebalance","pair_lookback","pair_entry_z","sharpe"]]
    bot = valid.nsmallest(3, "sharpe")[["cs_lookback","cs_rebalance","pair_lookback","pair_entry_z","sharpe"]]
    lines.append("  Top 3:")
    for _, r in top.iterrows():
        lines.append(f"    cs_lb={int(r['cs_lookback'])} cs_rb={int(r['cs_rebalance'])} "
                     f"pair_lb={int(r['pair_lookback'])} entry_z={r['pair_entry_z']:.1f} "
                     f"Sharpe={r['sharpe']:+.3f}")
    lines.append("  Bottom 3:")
    for _, r in bot.iterrows():
        lines.append(f"    cs_lb={int(r['cs_lookback'])} cs_rb={int(r['cs_rebalance'])} "
                     f"pair_lb={int(r['pair_lookback'])} entry_z={r['pair_entry_z']:.1f} "
                     f"Sharpe={r['sharpe']:+.3f}")
    return "\n".join(lines)


def walk_forward(prices: pd.DataFrame, n_windows: int = 3) -> None:
    """Anchored walk-forward with joint parameter selection."""
    # Reduced grid for speed.
    grid = []
    for cs_lb in [21, 42, 84]:
        for pair_lb in [21, 42]:
            for pair_entry in [1.5, 2.0]:
                grid.append({
                    "cs_lookback": cs_lb, "cs_n_long": 3, "cs_n_short": 3,
                    "cs_rebalance": 6,
                    "pair_lookback": pair_lb, "pair_entry_z": pair_entry,
                    "pair_exit_z": 0.5,
                })

    start = prices.index.min()
    end = prices.index.max()
    boundaries = pd.date_range(start, end, periods=n_windows + 1, tz="UTC")

    compounded = 1.0
    n_total = 0
    n_profitable = 0

    for i in range(n_windows - 1):
        test_start = boundaries[i + 1]
        test_end = boundaries[i + 2] if i + 2 < len(boundaries) else end + pd.Timedelta("1s")

        train_prices = prices.loc[prices.index < test_start]
        test_prices = prices.loc[(prices.index >= test_start) & (prices.index < test_end)]

        if len(train_prices) < 500 or len(test_prices) < 200:
            print(f"    window {i}: skipped (insufficient data)")
            continue

        # Pick best params on training window.
        best_sharpe = -1e9
        best_params = None
        for params in grid:
            try:
                r = _run(train_prices, params)
                s = float(r.metrics.get("sharpe", float("nan")))
                if not np.isnan(s) and s > best_sharpe:
                    best_sharpe = s
                    best_params = params
            except Exception:
                pass

        if best_params is None:
            continue

        # Test on unseen window, with warmup from training tail.
        warmup = train_prices.tail(300)
        combined = pd.concat([warmup, test_prices])
        r = _run(combined, best_params)
        eq = r.equity
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

        print(
            f"    {test_start.date()} -> {test_end.date()} | "
            f"chosen: cs_lb={best_params['cs_lookback']} "
            f"pair_lb={best_params['pair_lookback']} "
            f"entry_z={best_params['pair_entry_z']} | "
            f"train Sh={best_sharpe:+.2f} test Sh={sh:+.2f} ret={ret:+.1%}"
        )

    print(f"  windows:               {n_total}")
    print(f"  compounded OOS return: {compounded - 1.0:+.2%}")
    print(f"  windows profitable:    {n_profitable}/{n_total}")


def quarterly_breakdown(prices: pd.DataFrame, params: dict) -> None:
    r = _run(prices, params)
    eq = r.equity
    rets = eq.pct_change().fillna(0.0)
    rets.index = pd.to_datetime(rets.index, utc=True)

    grouped = rets.groupby(pd.Grouper(freq="QE"))
    print(f"  {'quarter_end':>13s}  {'return':>10s}  {'sharpe':>8s}  {'bars':>5s}")
    for q, s in grouped:
        if len(s) < 50:
            continue
        total = float((1 + s).prod() - 1)
        std = float(s.std())
        sh = float(s.mean() / std * np.sqrt(PERIODS_PER_YEAR)) if std > 0 else float("nan")
        print(f"  {str(q.date()):>13s}  {total:>+9.2%}  {sh:>+8.2f}  {len(s):>5d}")


def main() -> None:
    with open("config/settings.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)["data"]
    prices = _load_all(cfg["symbols"], START, END)

    days = (prices.index[-1] - prices.index[0]).days
    years = days / 365.25

    print(f"\n{'=' * 82}")
    print(f"  Combined Reversion Portfolio — Full Validation")
    print(f"  {prices.shape[1]} symbols | {prices.shape[0]} bars | ~{years:.2f} years")
    print(f"{'=' * 82}")

    # [1] Parameter sweep
    print("\n[1] Joint parameter sweep (54 configs: cs_lb × cs_rb × pair_lb × pair_entry)")
    sweep = parameter_sweep(prices)
    print(summarize_sweep(sweep))
    sweep.to_csv("research/combined_reversion_sweep.csv", index=False)

    # [2] Walk-forward
    print("\n[2] Walk-forward (anchored, 3 windows, joint param selection)")
    walk_forward(prices, n_windows=3)

    # [3] Quarterly
    print("\n[3] Quarterly breakdown (chosen params)")
    quarterly_breakdown(prices, CHOSEN)

    # [4] Bootstrap
    print("\n[4] Bootstrap on chosen-params return series")
    r = _run(prices, CHOSEN)
    bs = block_bootstrap_sharpe(r.returns, periods_per_year=PERIODS_PER_YEAR)
    print(f"  Sharpe point:    {bs['sharpe_point']:+.3f}")
    print(f"  5th percentile:  {bs['p05']:+.3f}")
    print(f"  50th percentile: {bs['p50']:+.3f}")
    print(f"  95th percentile: {bs['p95']:+.3f}")
    print(f"  P(Sharpe > 0):   {bs['pct_above_zero']:.1%}")

    # [5] Cost sensitivity
    print("\n[5] Cost sensitivity on chosen params")
    print(f"  {'bps/side':>9s}  {'Sharpe':>8s}  {'TotRet':>10s}  {'MaxDD':>9s}")
    for bps in [5.0, 10.0, 15.0, 25.0]:
        r = _run(prices, CHOSEN, cost_bps=bps)
        m = r.metrics
        print(f"  {bps:>9.1f}  {m['sharpe']:>+8.2f}  {m['total_return']:>+9.1%}  {m['max_drawdown']:>+9.2%}")


if __name__ == "__main__":
    main()
    
"""Validation framework applied to TSMOM (LAB-001).

Runs: 1D parameter sweep, walk-forward, regime breakdown, bootstrap.
TSMOM has only one parameter (lookback_n), so the sweep is 1D.

Run:
    uv run python scripts/validate_tsmom.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from crypto_algo.backtesting.simulator import BacktestResult, run_backtest
from crypto_algo.data.loaders import load_parquet
from crypto_algo.strategies.tsmom import tsmom_target
from crypto_algo.validation.bootstrap import (
    block_bootstrap_sharpe,
    format_bootstrap_output,
    trade_sign_test,
)
from crypto_algo.validation.regime import (
    calendar_year_breakdown,
    format_regime_output,
    volatility_regime_breakdown,
)

START = "2020-01-01"
END = "2025-01-01"
SYMBOLS = ["BTCUSDT", "ETHUSDT"]
INITIAL_EQUITY = 10_000.0
PERIODS_PER_YEAR = 6 * 365
FUTURES_TAKER_BPS = 5.0

LOOKBACK_GRID = [14, 21, 30, 42, 60, 84, 120, 168]
CHOSEN_LOOKBACK = 42


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


COST = _Cost(FUTURES_TAKER_BPS)


def _load(symbol: str) -> pd.DataFrame:
    return load_parquet(
        Path(f"data/raw/binance/futures/{symbol}/4h/{START}_{END}.parquet")
    )


def _run_one(df: pd.DataFrame, lookback: int) -> BacktestResult:
    target = tsmom_target(df["close"], lookback_n=lookback).reset_index(drop=True)
    target.index = df.index
    return run_backtest(df, target, COST, INITIAL_EQUITY, PERIODS_PER_YEAR)


def _sweep(df: pd.DataFrame, lookbacks: list[int]) -> pd.DataFrame:
    rows = []
    for lb in lookbacks:
        try:
            r = _run_one(df, lb)
            row = {"lookback": lb}
            row.update(r.metrics)
            rows.append(row)
        except Exception as exc:  # noqa: BLE001
            logger.warning("sweep failed for lookback {}: {}", lb, exc)
    return pd.DataFrame(rows)


def _summarize_sweep(sweep: pd.DataFrame, chosen: int) -> str:
    if sweep.empty:
        return "Empty sweep."
    valid = sweep.dropna(subset=["sharpe"]).copy()
    n = len(valid)
    n_pos = int((valid["sharpe"] > 0).sum())
    n_gt_05 = int((valid["sharpe"] > 0.5).sum())
    n_gt_10 = int((valid["sharpe"] > 1.0).sum())
    median_sharpe = float(valid["sharpe"].median())
    median_trades = float(valid["n_trades"].median())
    best_row = valid.loc[valid["sharpe"].idxmax()]
    worst_row = valid.loc[valid["sharpe"].idxmin()]

    chosen_row = valid[valid["lookback"] == chosen]
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
        f"  median Sharpe:          {median_sharpe:.3f}",
        f"  median trades:          {median_trades:.0f}",
        f"  best:  lookback={int(best_row['lookback'])} Sharpe={best_row['sharpe']:.3f}",
        f"  worst: lookback={int(worst_row['lookback'])} Sharpe={worst_row['sharpe']:.3f}",
        f"  chosen: lookback={chosen} Sharpe={chosen_sharpe:.3f} (pct {chosen_pct:.0f})",
    ]

    if n_gt_05 / n > 0.6:
        lines.append("  VERDICT: broad region of decent Sharpe. Consistent with real edge.")
    elif n_pos / n < 0.4:
        lines.append("  VERDICT: most of grid loses. Chosen lookback may be lucky.")
    else:
        lines.append("  VERDICT: mixed. Inconclusive.")
    return "\n".join(lines)


def _walk_forward(df: pd.DataFrame, lookbacks: list[int], n_windows: int = 5) -> None:
    start = df["open_time"].min()
    end = df["open_time"].max()
    boundaries = pd.date_range(start, end, periods=n_windows + 1, tz="UTC")

    compounded = 1.0
    n_total = 0
    n_profitable = 0
    test_sharpes = []

    for i in range(n_windows - 1):
        test_start = boundaries[i + 1]
        test_end = boundaries[i + 2] if i + 2 < len(boundaries) else end + pd.Timedelta("1s")

        train_df = df[df["open_time"] < test_start].reset_index(drop=True)
        if len(train_df) < 500:
            continue

        train_sweep = _sweep(train_df, lookbacks)
        if train_sweep.empty:
            continue
        best = train_sweep.loc[train_sweep["sharpe"].idxmax()]
        chosen_lb = int(best["lookback"])

        test_df = df[(df["open_time"] >= test_start) & (df["open_time"] < test_end)].reset_index(drop=True)
        if len(test_df) < 100:
            continue

        test_first_time = test_df["open_time"].iloc[0]
        train_tail = df[df["open_time"] < test_first_time].tail(200).reset_index(drop=True)
        combined = pd.concat([train_tail, test_df], ignore_index=True)

        r = _run_one(combined, chosen_lb)
        mask = r.equity.index >= test_start
        eq = r.equity[mask]
        if len(eq) < 50:
            continue
        eq = eq / eq.iloc[0] * INITIAL_EQUITY
        ret = float(eq.iloc[-1] / eq.iloc[0] - 1.0)
        rets = eq.pct_change().dropna()
        std = float(rets.std())
        sh = float(rets.mean() / std * np.sqrt(PERIODS_PER_YEAR)) if std > 0 else float("nan")

        compounded *= (1.0 + ret)
        n_total += 1
        if ret > 0:
            n_profitable += 1
        test_sharpes.append(sh)

        trades = r.trades[r.trades["time"] >= test_start] if len(r.trades) else r.trades
        print(
            f"    {test_start.date()} -> {test_end.date()} "
            f"chosen_lookback={chosen_lb} "
            f"train Sh={float(best['sharpe']):+.2f} "
            f"test Sh={sh:+.2f} ret={ret:+.1%} trades={int(len(trades))}"
        )

    if n_total == 0:
        print("  No valid walk-forward windows.")
        return

    print(f"  windows:               {n_total}")
    print(f"  compounded OOS return: {compounded - 1.0:+.2%}")
    print(f"  mean test Sharpe:      {float(np.mean(test_sharpes)):+.3f}")
    print(f"  windows profitable:    {n_profitable / n_total:.0%}")


def main() -> None:
    for symbol in SYMBOLS:
        df = _load(symbol)

        print(f"\n{'=' * 72}")
        print(f"  {symbol}  —  TSMOM  —  validation suite")
        print(f"{'=' * 72}")

        print("\n[1] Parameter sweep (lookback_n)")
        sweep = _sweep(df, LOOKBACK_GRID)
        print(_summarize_sweep(sweep, CHOSEN_LOOKBACK))

        print("\n[2] Walk-forward (anchored, expanding)")
        _walk_forward(df, LOOKBACK_GRID, n_windows=5)

        print("\n[3] Regime breakdown (chosen lookback=42)")
        r_chosen = _run_one(df, CHOSEN_LOOKBACK)
        bench = df.set_index("open_time")["close"].pct_change().fillna(0.0)
        bench.index = pd.to_datetime(bench.index, utc=True)
        cal = calendar_year_breakdown(r_chosen, bench, PERIODS_PER_YEAR)
        vol = volatility_regime_breakdown(r_chosen, bench, PERIODS_PER_YEAR)
        print(format_regime_output(cal, vol))

        print("\n[4] Statistical significance (chosen lookback=42)")
        bs = block_bootstrap_sharpe(r_chosen.returns, periods_per_year=PERIODS_PER_YEAR)
        sign = trade_sign_test(r_chosen.trades)
        print(format_bootstrap_output(bs, sign))


if __name__ == "__main__":
    main()
    
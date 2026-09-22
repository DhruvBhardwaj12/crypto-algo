"""Validation framework applied to Donchian + RSI multi-timeframe (EXP-004).

Runs: parameter sweep, walk-forward, regime breakdown, bootstrap.
Custom summarizer because our parameters are (dc, dc_exit, rsi4h).

Run:
    uv run python scripts/validate_donchian_rsi.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from crypto_algo.backtesting.simulator import BacktestResult, run_backtest
from crypto_algo.data.loaders import load_parquet
from crypto_algo.strategies.donchian_rsi import (
    donchian_rsi_target,
    tag_with_4h_regime,
)
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
PERIODS_PER_YEAR = 24 * 365
FUTURES_TAKER_BPS = 5.0


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


def _load(symbol: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    df_1h = load_parquet(
        Path(f"data/raw/binance/futures/{symbol}/1h/{START}_{END}.parquet")
    )
    df_4h = load_parquet(
        Path(f"data/raw/binance/futures/{symbol}/4h/{START}_{END}.parquet")
    )
    return df_1h, df_4h


def _build_target(
    df_1h: pd.DataFrame, rsi_4h: pd.Series, params: dict
) -> pd.Series:
    return donchian_rsi_target(
        df_1h,
        rsi_4h,
        donchian_entry_lookback=int(params["dc"]),
        donchian_exit_lookback=int(params["dc_exit"]),
        rsi_1h_period=14,
        rsi_1h_entry=50.0,
        rsi_4h_entry=float(params["rsi4h"]),
        rsi_4h_exit=48.0,
    )


def _run_one(
    df_1h: pd.DataFrame,
    rsi_4h: pd.Series,
    params: dict,
    periods_per_year: int = PERIODS_PER_YEAR,
) -> BacktestResult:
    target = _build_target(df_1h, rsi_4h, params).reset_index(drop=True)
    target.index = df_1h.index
    return run_backtest(df_1h, target, COST, INITIAL_EQUITY, periods_per_year)


def _sweep(
    df_1h: pd.DataFrame, rsi_4h: pd.Series, grid: list[dict]
) -> pd.DataFrame:
    rows = []
    for params in grid:
        try:
            r = _run_one(df_1h, rsi_4h, params)
            row = dict(params)
            row.update(r.metrics)
            rows.append(row)
        except Exception as exc:  # noqa: BLE001
            logger.warning("sweep param failed: {} ({})", params, exc)
    return pd.DataFrame(rows)


def _summarize_sweep(sweep: pd.DataFrame, chosen: dict) -> str:
    if sweep.empty:
        return "Empty sweep."
    valid = sweep.dropna(subset=["sharpe"]).copy()
    if valid.empty:
        return "No valid Sharpe values."

    n = len(valid)
    n_pos = int((valid["sharpe"] > 0).sum())
    n_gt_05 = int((valid["sharpe"] > 0.5).sum())
    n_gt_10 = int((valid["sharpe"] > 1.0).sum())
    median_sharpe = float(valid["sharpe"].median())
    median_trades = float(valid["n_trades"].median())
    best_row = valid.loc[valid["sharpe"].idxmax()]
    worst_row = valid.loc[valid["sharpe"].idxmin()]

    chosen_row = valid[
        (valid["dc"] == chosen["dc"])
        & (valid["dc_exit"] == chosen["dc_exit"])
        & (valid["rsi4h"] == chosen["rsi4h"])
    ]
    chosen_sharpe = (
        float(chosen_row["sharpe"].iloc[0]) if not chosen_row.empty else float("nan")
    )
    chosen_pct = (
        float((valid["sharpe"] < chosen_sharpe).mean() * 100)
        if not chosen_row.empty
        else float("nan")
    )

    lines = [
        f"  grid size:              {n}",
        f"  Sharpe > 0:             {n_pos}/{n} ({n_pos / n:.0%})",
        f"  Sharpe > 0.5:           {n_gt_05}/{n} ({n_gt_05 / n:.0%})",
        f"  Sharpe > 1.0:           {n_gt_10}/{n} ({n_gt_10 / n:.0%})",
        f"  median Sharpe:          {median_sharpe:.3f}",
        f"  median trades:          {median_trades:.0f}",
        f"  best:   dc={int(best_row['dc'])} dc_exit={int(best_row['dc_exit'])} "
        f"rsi4h={best_row['rsi4h']:.0f} Sharpe={best_row['sharpe']:.3f}",
        f"  worst:  dc={int(worst_row['dc'])} dc_exit={int(worst_row['dc_exit'])} "
        f"rsi4h={worst_row['rsi4h']:.0f} Sharpe={worst_row['sharpe']:.3f}",
        f"  chosen: dc={chosen['dc']} dc_exit={chosen['dc_exit']} rsi4h={chosen['rsi4h']:.0f} "
        f"Sharpe={chosen_sharpe:.3f} (pct {chosen_pct:.0f})",
    ]

    if n_gt_05 / n > 0.6:
        lines.append("  VERDICT: broad region of decent Sharpe. Consistent with real edge.")
    elif n_pos / n < 0.4:
        lines.append("  VERDICT: most of grid loses. Chosen params may be a lucky spot.")
    else:
        lines.append("  VERDICT: mixed. Inconclusive.")

    return "\n".join(lines)


def _walk_forward(
    df_1h: pd.DataFrame,
    rsi_4h: pd.Series,
    grid: list[dict],
    n_windows: int = 5,
    warmup_bars: int = 200,
) -> None:
    """Simple anchored walk-forward for a 3-param strategy."""
    start = df_1h["open_time"].min()
    end = df_1h["open_time"].max()
    boundaries = pd.date_range(start, end, periods=n_windows + 1, tz="UTC")

    compounded = 1.0
    n_profitable = 0
    n_total = 0
    test_sharpes = []

    for i in range(n_windows - 1):
        test_start = boundaries[i + 1]
        test_end = boundaries[i + 2] if i + 2 < len(boundaries) else end + pd.Timedelta("1s")

        train_mask = df_1h["open_time"] < test_start
        train_df = df_1h[train_mask].reset_index(drop=True)
        rsi_4h_train = rsi_4h[train_mask.values].reset_index(drop=True)

        if len(train_df) < 3000:
            logger.warning("Window {} skipped (train too small).", i)
            continue

        train_sweep = _sweep(train_df, rsi_4h_train, grid)
        if train_sweep.empty:
            continue
        best = train_sweep.loc[train_sweep["sharpe"].idxmax()]
        chosen_w = {
            "dc": int(best["dc"]),
            "dc_exit": int(best["dc_exit"]),
            "rsi4h": float(best["rsi4h"]),
        }

        # Build combined df: warmup from train tail + test window
        test_mask = (df_1h["open_time"] >= test_start) & (df_1h["open_time"] < test_end)
        test_df = df_1h[test_mask].reset_index(drop=True)
        if len(test_df) < 200:
            continue

        test_first_time = test_df["open_time"].iloc[0]
        train_tail_mask = df_1h["open_time"] < test_first_time
        train_tail = df_1h[train_tail_mask].tail(warmup_bars).reset_index(drop=True)
        combined_1h = pd.concat([train_tail, test_df], ignore_index=True)

        # Combined rsi_4h: use the 4H feature at the merged 1H times.
        # Since rsi_4h is a Series aligned to df_1h's original index, we need
        # to recompute the merge on the combined frame.
        # Simpler approach: rsi_4h values follow the same rule (tag with last
        # closed 4H bar), so we can index by open_time.
        rsi_4h_by_time = pd.Series(
            rsi_4h.values, index=df_1h["open_time"].values
        )
        combined_rsi_4h = rsi_4h_by_time.reindex(
            combined_1h["open_time"].values
        ).reset_index(drop=True)

        r = _run_one(combined_1h, combined_rsi_4h, chosen_w)

        mask = r.equity.index >= test_start
        eq = r.equity[mask]
        if len(eq) < 100:
            continue
        eq = eq / eq.iloc[0] * INITIAL_EQUITY
        ret = float(eq.iloc[-1] / eq.iloc[0] - 1.0)
        rets = eq.pct_change().dropna()
        std = float(rets.std())
        sh = (
            float(rets.mean() / std * np.sqrt(PERIODS_PER_YEAR))
            if std > 0
            else float("nan")
        )

        compounded *= (1.0 + ret)
        n_total += 1
        if ret > 0:
            n_profitable += 1
        test_sharpes.append(sh)

        # Count trades in test window.
        trades = r.trades[r.trades["time"] >= test_start] if len(r.trades) else r.trades
        n_trades_test = int(len(trades))

        print(
            f"    {test_start.date()} -> {test_end.date()} "
            f"chosen=({chosen_w['dc']},{chosen_w['dc_exit']},{chosen_w['rsi4h']:.0f}) "
            f"train Sh={float(best['sharpe']):+.2f} "
            f"test Sh={sh:+.2f} ret={ret:+.1%} trades={n_trades_test}"
        )

    if n_total == 0:
        print("  No valid walk-forward windows.")
        return

    compounded_ret = compounded - 1.0
    mean_test_sharpe = float(np.mean(test_sharpes)) if test_sharpes else float("nan")
    pct_pos = n_profitable / n_total

    print(f"  windows:               {n_total}")
    print(f"  compounded OOS return: {compounded_ret:+.2%}")
    print(f"  mean test Sharpe:      {mean_test_sharpe:+.3f}")
    print(f"  windows profitable:    {pct_pos:.0%}")


def main() -> None:
    grid = [
        {"dc": dc, "dc_exit": ex, "rsi4h": r}
        for dc in [40, 55, 75]
        for ex in [15, 20, 30]
        for r in [50.0, 52.0, 55.0]
    ]
    chosen = {"dc": 55, "dc_exit": 20, "rsi4h": 52.0}

    for symbol in SYMBOLS:
        df_1h, df_4h = _load(symbol)
        rsi_4h = tag_with_4h_regime(df_1h, df_4h, rsi_period=14)

        print(f"\n{'=' * 72}")
        print(f"  {symbol}  —  Donchian + RSI  —  validation suite")
        print(f"{'=' * 72}")

        # [1] Parameter sweep
        print("\n[1] Parameter sweep (dc × dc_exit × rsi4h)")
        sweep = _sweep(df_1h, rsi_4h, grid)
        print(_summarize_sweep(sweep, chosen))

        # [2] Walk-forward
        print("\n[2] Walk-forward (anchored, expanding)")
        _walk_forward(df_1h, rsi_4h, grid, n_windows=5, warmup_bars=200)

        # [3] Regime
        print("\n[3] Regime breakdown (chosen params)")
        r_chosen = _run_one(df_1h, rsi_4h, chosen)
        bench = df_1h.set_index("open_time")["close"].pct_change().fillna(0.0)
        bench.index = pd.to_datetime(bench.index, utc=True)
        cal = calendar_year_breakdown(r_chosen, bench, PERIODS_PER_YEAR)
        vol = volatility_regime_breakdown(r_chosen, bench, PERIODS_PER_YEAR)
        print(format_regime_output(cal, vol))

        # [4] Bootstrap
        print("\n[4] Statistical significance (chosen params)")
        bs = block_bootstrap_sharpe(
            r_chosen.returns, periods_per_year=PERIODS_PER_YEAR
        )
        sign = trade_sign_test(r_chosen.trades)
        print(format_bootstrap_output(bs, sign))


if __name__ == "__main__":
    main()
    
"""Walk-forward analysis.

We split the timeline into N+1 segments. On each step:
  - train on everything before the test segment
  - sweep params on train, pick best by Sharpe
  - run the chosen params on the test segment (with warmup from train)
  - record the test result

Only the aggregate of test results matters. If a strategy's edge is real,
picking parameters on past data should produce positive returns on the
following, unseen period. If it doesn't, the edge is illusory.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from loguru import logger

from crypto_algo.backtesting.costs import CostModel
from crypto_algo.backtesting.simulator import compute_metrics
from crypto_algo.validation.common import StrategyFn, run_strategy
from crypto_algo.validation.parameter_sweep import parameter_sweep


@dataclass
class WindowResult:
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    chosen_params: dict
    train_sharpe: float
    test_sharpe: float
    test_return: float
    test_max_dd: float
    n_trades: int


def walk_forward(
    df: pd.DataFrame,
    strategy_fn: StrategyFn,
    param_grid: list[dict],
    n_windows: int,
    cost_model: CostModel,
    initial_equity: float,
    periods_per_year: int,
    warmup_bars: int = 110,
) -> dict:
    """Run anchored (expanding-window) walk-forward. Returns aggregate dict."""
    if n_windows < 2:
        raise ValueError("n_windows must be >= 2")

    start = df["open_time"].min()
    end = df["open_time"].max()
    # n_windows test segments; boundaries has n_windows+1 points.
    boundaries = pd.date_range(start, end, periods=n_windows + 1, tz="UTC")

    window_results: list[WindowResult] = []

    for i in range(n_windows - 1):
        test_start = boundaries[i + 1]
        test_end = boundaries[i + 2] if i + 2 < len(boundaries) else end + pd.Timedelta("1s")

        train_df = df[df["open_time"] < test_start].copy()
        test_df = df[(df["open_time"] >= test_start) & (df["open_time"] < test_end)].copy()

        if len(train_df) < 200 or len(test_df) < 30:
            logger.warning("Skipping window {} (too small).", i)
            continue

        # Pick params by in-sample Sharpe on training data.
        train_sweep = parameter_sweep(
            train_df, strategy_fn, param_grid, cost_model, initial_equity, periods_per_year
        )
        if train_sweep.empty:
            continue
        best_idx = train_sweep["sharpe"].idxmax()
        best_row = train_sweep.loc[best_idx]
        chosen = {k: int(best_row[k]) for k in param_grid[0].keys() if k in best_row}
        train_sharpe = float(best_row["sharpe"])

        # Build test df with warmup from the tail of training.
        warmup = train_df.tail(warmup_bars)
        combined = pd.concat([warmup, test_df], ignore_index=True)

        result = run_strategy(
            combined, strategy_fn, chosen, cost_model, initial_equity, periods_per_year
        )

        # Slice result to test window and rebase equity.
        mask = result.equity.index >= test_start
        equity = result.equity[mask]
        if len(equity) < 10:
            continue
        equity = equity / equity.iloc[0] * initial_equity

        trades = result.trades[result.trades["time"] >= test_start] if len(result.trades) else result.trades
        metrics = compute_metrics(equity, trades, periods_per_year)

        window_results.append(
            WindowResult(
                test_start=test_start,
                test_end=test_end,
                chosen_params=chosen,
                train_sharpe=train_sharpe,
                test_sharpe=float(metrics.get("sharpe", float("nan"))),
                test_return=float(equity.iloc[-1] / equity.iloc[0] - 1.0),
                test_max_dd=float(metrics.get("max_drawdown", float("nan"))),
                n_trades=int(metrics.get("n_trades", 0)),
            )
        )

    if not window_results:
        return {"windows": [], "summary": "No valid windows."}

    df_windows = pd.DataFrame([w.__dict__ for w in window_results])

    # Aggregate: geometric chain of test-window returns.
    compounded = (1.0 + df_windows["test_return"]).prod() - 1.0
    mean_test_sharpe = float(df_windows["test_sharpe"].mean())
    pct_positive = float((df_windows["test_return"] > 0).mean())

    summary = "\n".join([
        f"  windows:               {len(window_results)}",
        f"  compounded OOS return: {compounded:+.2%}",
        f"  mean test Sharpe:      {mean_test_sharpe:.3f}",
        f"  windows profitable:    {pct_positive:.0%}",
        "  per-window:",
    ] + [
        f"    {w.test_start.date()} -> {w.test_end.date()} "
        f"params=({w.chosen_params.get('fast')},{w.chosen_params.get('slow')}) "
        f"train Sharpe={w.train_sharpe:+.2f} "
        f"test Sharpe={w.test_sharpe:+.2f} "
        f"ret={w.test_return:+.1%} "
        f"trades={w.n_trades}"
        for w in window_results
    ])

    return {
        "windows": window_results,
        "df": df_windows,
        "compounded_return": compounded,
        "mean_test_sharpe": mean_test_sharpe,
        "pct_positive": pct_positive,
        "summary": summary,
    }


    
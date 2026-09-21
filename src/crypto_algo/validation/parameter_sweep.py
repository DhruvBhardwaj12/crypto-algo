"""Parameter stability sweep.

If a strategy only works at ONE magic parameter pair, it's a coincidence.
If it works across a broad, smooth region of the parameter space, it
reflects something more like a real edge.

We do NOT use this to pick the best params. We use it to test whether the
chosen params sit inside a stable region or on an isolated spike.
"""

from __future__ import annotations

from itertools import product

import pandas as pd
from loguru import logger

from crypto_algo.backtesting.costs import CostModel
from crypto_algo.validation.common import StrategyFn, run_strategy


def build_grid(fast_range: list[int], slow_range: list[int]) -> list[dict]:
    """All (fast, slow) pairs with fast < slow."""
    return [
        {"fast": f, "slow": s}
        for f, s in product(fast_range, slow_range)
        if f < s
    ]


def parameter_sweep(
    df: pd.DataFrame,
    strategy_fn: StrategyFn,
    param_grid: list[dict],
    cost_model: CostModel,
    initial_equity: float,
    periods_per_year: int,
) -> pd.DataFrame:
    """Run strategy over every param set. Returns one row per set."""
    rows: list[dict] = []
    for i, params in enumerate(param_grid, 1):
        try:
            result = run_strategy(
                df, strategy_fn, params, cost_model, initial_equity, periods_per_year
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Param set {} failed: {}", params, exc)
            continue
        row = dict(params)
        row.update(result.metrics)
        rows.append(row)
        if i % 10 == 0:
            logger.info("  swept {}/{}", i, len(param_grid))
    return pd.DataFrame(rows)


def summarize_sweep(sweep: pd.DataFrame, chosen: dict) -> str:
    """Print a short text summary of a parameter sweep."""
    if sweep.empty:
        return "Empty sweep."

    valid = sweep.dropna(subset=["sharpe"])
    if valid.empty:
        return "No valid Sharpe values in sweep."

    n = len(valid)
    n_pos = int((valid["sharpe"] > 0).sum())
    n_gt_05 = int((valid["sharpe"] > 0.5).sum())
    median_sharpe = float(valid["sharpe"].median())
    best_row = valid.loc[valid["sharpe"].idxmax()]
    worst_row = valid.loc[valid["sharpe"].idxmin()]

    chosen_row = valid[
        (valid["fast"] == chosen["fast"]) & (valid["slow"] == chosen["slow"])
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
        f"  median Sharpe:          {median_sharpe:.3f}",
        f"  best:   fast={int(best_row['fast'])} slow={int(best_row['slow'])} "
        f"Sharpe={best_row['sharpe']:.3f}",
        f"  worst:  fast={int(worst_row['fast'])} slow={int(worst_row['slow'])} "
        f"Sharpe={worst_row['sharpe']:.3f}",
        f"  chosen: fast={chosen['fast']} slow={chosen['slow']} "
        f"Sharpe={chosen_sharpe:.3f} (percentile {chosen_pct:.0f})",
    ]

    # Interpretation hints.
    if n_pos / n < 0.4:
        lines.append("  VERDICT: most of the grid loses. Chosen params may be a lucky spot.")
    elif n_gt_05 / n > 0.5:
        lines.append("  VERDICT: broad region of positive Sharpe. Consistent with real edge.")
    else:
        lines.append("  VERDICT: mixed. Inconclusive.")

    return "\n".join(lines)

    
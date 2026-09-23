"""Portfolio-level backtest for cross-sectional (long-short) strategies.

Unlike the single-symbol simulator, this operates on a WIDE price matrix
(rows = time, columns = symbols) and a matching WIDE weight matrix.

Weights can be negative (shorts). Gross exposure typically sums to 1.0
(e.g., +1/(2K) across K longs and -1/(2K) across K shorts).

Convention:
- weights[i] is the target portfolio at close of bar i.
- Execution happens at bar i+1's open. We model this by lagging weights.
- P&L during bar i = sum(weights_prev * return_i), then pay costs on the
  weight change.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class PortfolioResult:
    equity: pd.Series
    weights: pd.DataFrame
    returns: pd.Series
    metrics: dict = field(default_factory=dict)


def _compute_metrics(equity: pd.Series, periods_per_year: int) -> dict:
    if len(equity) < 2:
        return {}
    rets = equity.pct_change().dropna()
    total_return = float(equity.iloc[-1] / equity.iloc[0] - 1.0)
    years = len(equity) / periods_per_year
    cagr = (
        float((equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1)
        if years > 0 and equity.iloc[-1] > 0
        else float("nan")
    )
    ann = np.sqrt(periods_per_year)
    mean_r = float(rets.mean())
    std_r = float(rets.std())
    sharpe = mean_r / std_r * ann if std_r > 0 else float("nan")

    downside = rets[rets < 0]
    dstd = float(downside.std()) if len(downside) > 1 else float("nan")
    sortino = mean_r / dstd * ann if dstd and dstd > 0 else float("nan")

    peak = equity.cummax()
    dd = equity / peak - 1.0
    max_dd = float(dd.min())
    calmar = cagr / abs(max_dd) if max_dd < 0 else float("nan")

    # Turnover: mean absolute weight change per bar.
    return {
        "total_return": total_return,
        "cagr": cagr,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_dd,
        "calmar": calmar,
    }


def run_portfolio_backtest(
    prices: pd.DataFrame,
    weights: pd.DataFrame,
    cost_model,
    initial_equity: float = 10_000.0,
    periods_per_year: int = 6 * 365,
) -> PortfolioResult:
    """Run a portfolio-level backtest.

    Parameters
    ----------
    prices : DataFrame of close prices, index=time, columns=symbols.
    weights : DataFrame of target weights, same shape. Values in [-1, 1].
    cost_model : object with .cost_fraction("buy"|"sell") -> fraction.
    """
    if prices.shape != weights.shape:
        raise ValueError(f"Shape mismatch: prices {prices.shape} vs weights {weights.shape}")
    if not prices.index.equals(weights.index):
        raise ValueError("prices and weights must have identical indices")

    # Lag weights by 1 bar to enforce causality: decision at close of bar i-1,
    # execution at open of bar i, hold through bar i.
    actual_weights = weights.shift(1).fillna(0.0)

    # Zero out weights for symbols without prices.
    actual_weights = actual_weights.where(prices.notna(), 0.0)

    # Close-to-close returns; NaN when either endpoint missing.
    rets = prices.pct_change()
    rets = rets.fillna(0.0)
    # Any NaN close should result in zero return (no price move we can attribute).
    rets = rets.where(prices.notna(), 0.0)

    n = len(prices)
    equity = np.zeros(n, dtype=float)
    equity[0] = initial_equity

    cost_buy = cost_model.cost_fraction("buy")
    cost_sell = cost_model.cost_fraction("sell")

    w_arr = actual_weights.to_numpy()
    r_arr = rets.to_numpy()

    for i in range(1, n):
        w_prev = w_arr[i - 1]
        w_curr = w_arr[i]
        delta = w_curr - w_prev

        # Cost of rebalancing from w_prev to w_curr at open of bar i.
        cost_frac = 0.0
        for d in delta:
            if d > 0:
                cost_frac += d * cost_buy
            elif d < 0:
                cost_frac += -d * cost_sell

        # Apply cost to equity first.
        eq_after_cost = equity[i - 1] * (1.0 - cost_frac)

        # Apply gross P&L from bar i's price move.
        gross_ret = float(np.dot(w_curr, r_arr[i]))
        equity[i] = eq_after_cost * (1.0 + gross_ret)

    equity_series = pd.Series(equity, index=prices.index, name="equity")
    returns = equity_series.pct_change().fillna(0.0)

    metrics = _compute_metrics(equity_series, periods_per_year)

    return PortfolioResult(
        equity=equity_series,
        weights=actual_weights,
        returns=returns,
        metrics=metrics,
    )

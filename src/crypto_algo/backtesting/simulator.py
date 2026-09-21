"""Minimal long-only, unlevered backtest simulator.

Conventions
-----------
- target_position[i] is the fraction of equity we WANT to hold during bar i.
  It must have been decided using only data available at the close of bar i-1.
- Execution happens at the OPEN of bar i.
- We hold through the bar (open to close) and mark to market at the close.
- Costs are paid on the traded notional at the moment of execution.

The strategy is responsible for producing a target_position series that is
correctly lagged (see ma_crossover.py for an example).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from crypto_algo.backtesting.costs import CostModel
from crypto_algo.data.schema import check_schema


@dataclass
class BacktestResult:
    equity: pd.Series
    position: pd.Series
    returns: pd.Series
    trades: pd.DataFrame
    initial_equity: float
    final_equity: float
    metrics: dict[str, float]


def _compute_metrics(
    equity: pd.Series,
    trades: pd.DataFrame,
    periods_per_year: int,
) -> dict[str, float]:
    rets = equity.pct_change().dropna()
    if len(rets) == 0:
        return {}

    total_return = float(equity.iloc[-1] / equity.iloc[0] - 1.0)
    years = len(equity) / periods_per_year
    cagr = float((equity.iloc[-1] / equity.iloc[0]) ** (1.0 / years) - 1.0) if years > 0 else float("nan")

    ann = np.sqrt(periods_per_year)
    mean_ret = float(rets.mean())
    std_ret = float(rets.std())
    sharpe = mean_ret / std_ret * ann if std_ret > 0 else float("nan")

    downside = rets[rets < 0]
    downside_std = float(downside.std()) if len(downside) > 1 else float("nan")
    sortino = mean_ret / downside_std * ann if downside_std and downside_std > 0 else float("nan")

    rolling_peak = equity.cummax()
    dd = equity / rolling_peak - 1.0
    max_dd = float(dd.min())
    calmar = cagr / abs(max_dd) if max_dd < 0 else float("nan")

    n_trades = int(len(trades))
    if n_trades > 0:
        gross_profit = float(trades.loc[trades["pnl"] > 0, "pnl"].sum())
        gross_loss = float(-trades.loc[trades["pnl"] < 0, "pnl"].sum())
        win_rate = float((trades["pnl"] > 0).mean())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")
    else:
        win_rate = float("nan")
        profit_factor = float("nan")

    return {
        "total_return": total_return,
        "cagr": cagr,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_dd,
        "calmar": calmar,
        "n_trades": n_trades,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
    }


def run_backtest(
    df: pd.DataFrame,
    target_position: pd.Series,
    cost_model: CostModel,
    initial_equity: float = 10_000.0,
    periods_per_year: int = 24 * 365,
) -> BacktestResult:
    check_schema(df)

    if not df["open_time"].is_monotonic_increasing:
        raise ValueError("df must be sorted ascending by open_time")
    if len(df) != len(target_position):
        raise ValueError(
            f"target_position length {len(target_position)} != df length {len(df)}"
        )
    tp = target_position.reset_index(drop=True)
    if tp.dropna().empty:
        raise ValueError("target_position is all NaN")
    if not tp.dropna().between(0.0, 1.0).all():
        raise ValueError("target_position values must lie in [0, 1]")

    df = df.reset_index(drop=True)

    opens = df["open"].to_numpy()
    closes = df["close"].to_numpy()
    times = df["open_time"].to_numpy()
    n = len(df)

    equity = np.empty(n, dtype=float)
    position = np.empty(n, dtype=float)
    trade_rows: list[dict] = []

    equity[0] = initial_equity
    position[0] = 0.0
    entry_equity = initial_equity
    entry_price: float | None = None

    for i in range(1, n):
        prev_equity = equity[i - 1]
        prev_pos = position[i - 1]
        desired = float(tp.iloc[i - 1])  # decision made at close of bar i-1

        # If desired is NaN (indicator not yet warmed up), stay flat.
        if np.isnan(desired):
            desired = 0.0

        delta = desired - prev_pos
        if abs(delta) > 1e-12:
            notional = abs(delta) * prev_equity
            side = "buy" if delta > 0 else "sell"
            cost = notional * cost_model.cost_fraction(side)
            prev_equity -= cost

            # Record trade P&L for win-rate stats (simple approximation).
            if prev_pos == 0.0 and desired > 0.0:
                entry_equity = prev_equity
                entry_price = opens[i]
            elif prev_pos > 0.0 and desired == 0.0 and entry_price is not None:
                pnl = prev_equity - entry_equity
                trade_rows.append({"time": times[i], "pnl": pnl})
                entry_equity = prev_equity
                entry_price = None

        bar_return = closes[i] / opens[i] - 1.0
        equity[i] = prev_equity * (1.0 + desired * bar_return)
        position[i] = desired

    equity_series = pd.Series(equity, index=df["open_time"], name="equity")
    position_series = pd.Series(position, index=df["open_time"], name="position")
    returns = equity_series.pct_change().fillna(0.0)
    trades_df = pd.DataFrame(trade_rows, columns=["time", "pnl"])

    metrics = _compute_metrics(equity_series, trades_df, periods_per_year)

    return BacktestResult(
        equity=equity_series,
        position=position_series,
        returns=returns,
        trades=trades_df,
        initial_equity=initial_equity,
        final_equity=float(equity[-1]),
        metrics=metrics,
    )


def buy_and_hold_equity(df: pd.DataFrame, initial_equity: float) -> pd.Series:
    """Buy-and-hold benchmark: buy at first open, hold to last close."""
    closes = df["close"].to_numpy()
    eq = initial_equity * closes / closes[0]
    return pd.Series(eq, index=df["open_time"], name="buy_and_hold")
    
    # Public alias so other modules can recompute metrics on a sliced equity series.
compute_metrics = _compute_metrics

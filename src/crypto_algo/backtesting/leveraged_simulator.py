"""Leveraged single-symbol backtest with liquidation modeling.

Handles long-only positions. target_position is 0 or 1 (signal strength).
Leverage multiplies notional. Liquidation is triggered when price drops
by 0.98/leverage from the entry price (approximates Binance maintenance
margin behavior).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class LeveragedResult:
    equity: pd.Series
    position: pd.Series
    returns: pd.Series
    n_trades: int
    n_liquidations: int
    initial_equity: float
    final_equity: float
    metrics: dict = field(default_factory=dict)


def _compute_metrics(
    equity: pd.Series, n_liquidations: int, n_trades: int, periods_per_year: int
) -> dict:
    if len(equity) < 2:
        return {}
    rets = equity.pct_change().dropna()
    total = float(equity.iloc[-1] / equity.iloc[0] - 1.0)
    years = len(equity) / periods_per_year
    cagr = (
        float((equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1)
        if years > 0 and equity.iloc[-1] > 0 else float("nan")
    )
    ann = np.sqrt(periods_per_year)
    std = float(rets.std())
    sharpe = float(rets.mean() / std * ann) if std > 0 else float("nan")

    downside = rets[rets < 0]
    dstd = float(downside.std()) if len(downside) > 1 else float("nan")
    sortino = float(rets.mean() / dstd * ann) if dstd and dstd > 0 else float("nan")

    peak = equity.cummax()
    dd = equity / peak - 1.0
    max_dd = float(dd.min())
    calmar = cagr / abs(max_dd) if max_dd < 0 else float("nan")

    # Monthly returns (approx 30 days per month, 4H bars)
    bars_per_month = 6 * 30
    if len(equity) > bars_per_month:
        monthly = equity.pct_change(bars_per_month).dropna()
        avg_monthly = float(monthly.mean())
        median_monthly = float(monthly.median())
        pct_positive_months = float((monthly > 0).mean())
    else:
        avg_monthly = float("nan")
        median_monthly = float("nan")
        pct_positive_months = float("nan")

    return {
        "total_return": total,
        "cagr": cagr,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_dd,
        "calmar": calmar,
        "avg_monthly_return": avg_monthly,
        "median_monthly_return": median_monthly,
        "pct_positive_months": pct_positive_months,
        "n_trades": n_trades,
        "n_liquidations": n_liquidations,
    }


def run_leveraged_backtest(
    df: pd.DataFrame,
    target_position: pd.Series,
    leverage: float,
    cost_model,
    initial_equity: float = 100.0,
    periods_per_year: int = 6 * 365,
) -> LeveragedResult:
    """Long-only leveraged backtest.

    target_position: values in {0, 1}. 1 means full position at leverage.
    """
    if leverage < 1:
        raise ValueError("leverage must be >= 1")
    if len(df) != len(target_position):
        raise ValueError("target_position length mismatch")

    df = df.reset_index(drop=True)
    tp = target_position.reset_index(drop=True).fillna(0).astype(int)

    opens = df["open"].to_numpy()
    closes = df["close"].to_numpy()
    lows = df["low"].to_numpy()
    times = df["open_time"].to_numpy()
    n = len(df)

    equity = np.zeros(n, dtype=float)
    position = np.zeros(n, dtype=float)
    equity[0] = initial_equity

    pos = 0.0           # notional weight (0 or leverage)
    entry_price = 0.0
    n_trades = 0
    n_liquidations = 0
    cf = cost_model.cost_fraction("buy")

    for i in range(1, n):
        prev_eq = equity[i - 1]
        desired = float(tp.iloc[i - 1]) * leverage

        # Execute at open of bar i if position changed
        if abs(desired - pos) > 1e-9:
            # Close old position if any
            if pos > 0:
                # We treat any change in position as a "rebalance" cost
                pass
            # Open/close cost proportional to change
            notional = abs(desired - pos) * prev_eq
            cost = notional * cf
            prev_eq -= cost
            if pos == 0 and desired > 0:
                entry_price = float(opens[i])
                n_trades += 1
            pos = desired

        if pos > 0:
            # Check liquidation within bar i
            liq_price = entry_price * (1.0 - 0.98 / leverage)
            if lows[i] <= liq_price:
                # Liquidated. Equity goes to approximately zero.
                equity[i] = 0.0
                position[i] = 0.0
                pos = 0.0
                n_liquidations += 1
                # From here on, no more positions (account is dead)
                if i + 1 < n:
                    equity[i + 1:] = 0.0
                break

            # Bar P&L (open to close)
            bar_ret = closes[i] / opens[i] - 1.0
            equity[i] = prev_eq * (1.0 + pos * bar_ret)
        else:
            equity[i] = prev_eq

        position[i] = pos

    eq_series = pd.Series(equity, index=df["open_time"], name="equity")
    pos_series = pd.Series(position, index=df["open_time"], name="position")
    rets = eq_series.pct_change().fillna(0.0)

    metrics = _compute_metrics(eq_series, n_liquidations, n_trades, periods_per_year)

    return LeveragedResult(
        equity=eq_series,
        position=pos_series,
        returns=rets,
        n_trades=n_trades,
        n_liquidations=n_liquidations,
        initial_equity=initial_equity,
        final_equity=float(equity[-1]),
        metrics=metrics,
    )

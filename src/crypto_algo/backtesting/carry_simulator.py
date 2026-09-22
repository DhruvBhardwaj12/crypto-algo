"""Simplified two-leg funding carry simulator.

Assumptions (all explicit — change this file if you want different ones):

1. We hold a spot long and a perp short of equal notional. Price P&L nets
   to approximately zero. We therefore model the position's P&L as funding
   income minus costs, with no price component.
2. Notional = equity when in position. No leverage applied to the funding
   stream. (In reality, more capital efficiency is possible with cross
   margin; ignoring this makes the backtest conservative.)
3. Entry cost is paid at the moment of entry. Exit cost at exit.
4. Funding is applied at every 8h timestamp where we are in position.
5. No basis risk between spot and perp (modeling this would only make
   results worse, so we ignore it here).
6. No liquidation modeling. (With delta-neutral, there is no directional
   liquidation risk, but a margin call on the short leg is possible if
   spot-perp divergence spikes. Deferred to a later milestone.)

The purpose of this simulator is to answer: does gross funding income,
minus round-trip costs, produce a positive net return under any
reasonable entry/exit rule? If not, the strategy is dead. If yes, we
build the full two-leg simulator with basis and margin modeling.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from crypto_algo.backtesting.futures_costs import FuturesCostModel


@dataclass
class CarryResult:
    equity: pd.Series
    in_position: pd.Series
    funding_paid: pd.Series   # per-event funding applied (fraction)
    funding_income: float     # cumulative dollar funding received
    n_entries: int
    n_exits: int
    total_costs: float
    initial_equity: float
    final_equity: float
    metrics: dict = field(default_factory=dict)


def _compute_metrics(
    equity: pd.Series,
    periods_per_year: int,
    n_entries: int,
    total_costs: float,
) -> dict:
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

    roll_peak = equity.cummax()
    dd = equity / roll_peak - 1.0
    max_dd = float(dd.min())

    return {
        "total_return": total_return,
        "cagr": cagr,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "n_entries": n_entries,
        "total_costs": total_costs,
    }


def run_carry(
    funding: pd.DataFrame,
    signal: pd.Series,
    cost_model: FuturesCostModel,
    initial_equity: float = 10_000.0,
    periods_per_year: int = 3 * 365,  # 3 funding events per day
) -> CarryResult:
    """Simulate carry strategy.

    Parameters
    ----------
    funding : DataFrame with 'funding_time' and 'funding_rate'.
    signal : Series aligned to funding rows. 1 = in position, 0 = out.
    """
    if not funding["funding_time"].is_monotonic_increasing:
        raise ValueError("funding must be sorted by funding_time")
    if len(funding) != len(signal):
        raise ValueError(
            f"signal length {len(signal)} != funding length {len(funding)}"
        )

    funding = funding.reset_index(drop=True)
    sig = signal.reset_index(drop=True).fillna(0).astype(int)
    if not sig.isin([0, 1]).all():
        raise ValueError("signal must be 0 or 1")

    rates = funding["funding_rate"].to_numpy()
    times = funding["funding_time"].to_numpy()
    n = len(funding)

    equity = np.empty(n, dtype=float)
    in_pos = np.zeros(n, dtype=np.int8)
    funding_paid = np.zeros(n, dtype=float)

    entry_cost_f = cost_model.entry_cost_fraction()
    exit_cost_f = cost_model.exit_cost_fraction()

    eq = initial_equity
    in_position = False
    funding_income_total = 0.0
    total_costs = 0.0
    n_entries = 0
    n_exits = 0

    for i in range(n):
        s = int(sig.iloc[i])

        if s == 1 and not in_position:
            cost = eq * entry_cost_f
            eq -= cost
            total_costs += cost
            in_position = True
            n_entries += 1

        if in_position:
            payment = eq * rates[i]
            eq += payment
            funding_paid[i] = rates[i]
            funding_income_total += payment

        if s == 0 and in_position:
            cost = eq * exit_cost_f
            eq -= cost
            total_costs += cost
            in_position = False
            n_exits += 1

        equity[i] = eq
        in_pos[i] = 1 if in_position else 0

    eq_series = pd.Series(equity, index=pd.to_datetime(times, utc=True), name="equity")
    pos_series = pd.Series(in_pos, index=eq_series.index, name="in_position")
    fund_series = pd.Series(funding_paid, index=eq_series.index, name="funding_rate_applied")

    metrics = _compute_metrics(eq_series, periods_per_year, n_entries, total_costs)

    return CarryResult(
        equity=eq_series,
        in_position=pos_series,
        funding_paid=fund_series,
        funding_income=funding_income_total,
        n_entries=n_entries,
        n_exits=n_exits,
        total_costs=total_costs,
        initial_equity=initial_equity,
        final_equity=float(equity[-1]),
        metrics=metrics,
    )

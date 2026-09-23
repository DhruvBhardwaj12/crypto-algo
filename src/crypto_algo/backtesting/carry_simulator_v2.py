"""Two-leg funding carry simulator with basis risk and mark-to-market.

Position: long 1 unit of spot + short 1 unit of perp (delta-neutral).

Every 8h funding event:
  - funding cash flow: if short perp, receive rate * notional (when rate > 0)
  - mark position to market: value = spot_value - (perp_entry - perp_now) - ...

P&L decomposition per funding interval:
  price_pnl = (spot_now - spot_entry) - (perp_now - perp_entry)
            = (spot_now - perp_now) - (spot_entry - perp_entry)
            = basis_change  (this is the crucial term)

  funding_pnl = cumulative funding received
  cost = entry_cost + exit_cost (paid at entry/exit)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from crypto_algo.backtesting.futures_costs import FuturesCostModel


@dataclass
class CarryV2Result:
    equity: pd.Series
    in_position: pd.Series
    funding_received: pd.Series
    basis_pnl: pd.Series
    n_entries: int
    total_funding: float
    total_basis: float
    total_costs: float
    initial_equity: float
    final_equity: float
    metrics: dict = field(default_factory=dict)


def _compute_metrics(equity: pd.Series, periods_per_year: int, n_entries: int) -> dict:
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

    # Monthly returns
    monthly = equity.resample("30D").last().pct_change().dropna()
    avg_monthly = float(monthly.mean()) if len(monthly) > 0 else float("nan")

    return {
        "total_return": total, "cagr": cagr, "sharpe": sharpe,
        "sortino": sortino, "max_drawdown": max_dd, "calmar": calmar,
        "avg_monthly": avg_monthly, "n_entries": n_entries,
    }


def run_carry_v2(
    spot: pd.DataFrame,
    perp: pd.DataFrame,
    funding: pd.DataFrame,
    signal: pd.Series,
    cost_model: FuturesCostModel,
    initial_equity: float = 10_000.0,
    periods_per_year: int = 3 * 365,
) -> CarryV2Result:
    """Simulate a long-spot / short-perp carry position.

    All three DataFrames must be aligned by timestamp, at funding-event
    frequency (8h). signal: 1 = hold position, 0 = flat.
    """
    n = len(funding)
    if not (len(spot) == len(perp) == len(funding) == len(signal)):
        raise ValueError("spot, perp, funding, signal must be same length")

    spot_c = spot["close"].to_numpy()
    perp_c = perp["close"].to_numpy()
    rates = funding["funding_rate"].to_numpy()
    times = pd.to_datetime(funding["funding_time"], utc=True)
    sig = signal.reset_index(drop=True).fillna(0).astype(int).to_numpy()

    equity = np.zeros(n, dtype=float)
    in_pos = np.zeros(n, dtype=np.int8)
    fund_series = np.zeros(n, dtype=float)
    basis_series = np.zeros(n, dtype=float)

    eq = initial_equity
    holding = False
    entry_spot = 0.0
    entry_perp = 0.0
    entry_basis = 0.0
    prev_spot = 0.0
    prev_perp = 0.0
    total_funding = 0.0
    total_basis = 0.0
    total_costs = 0.0
    n_entries = 0

    entry_cost_f = cost_model.entry_cost_fraction()
    exit_cost_f = cost_model.exit_cost_fraction()

    for i in range(n):
        s = int(sig[i])

        if s == 1 and not holding:
            # Open position at current prices.
            cost = eq * entry_cost_f
            eq -= cost
            total_costs += cost
            entry_spot = spot_c[i]
            entry_perp = perp_c[i]
            entry_basis = (entry_perp - entry_spot) / entry_spot
            prev_spot = entry_spot
            prev_perp = entry_perp
            holding = True
            n_entries += 1

        if holding:
            # Funding cash flow: short perp receives rate * notional when rate > 0.
            fund_payment = eq * rates[i]
            eq += fund_payment
            fund_series[i] = fund_payment
            total_funding += fund_payment

            # Basis P&L between current and previous event.
            curr_basis = (perp_c[i] - spot_c[i]) / spot_c[i]
            basis_pnl = (entry_basis - curr_basis) * eq  # short perp: basis widening hurts
            eq += basis_pnl
            basis_series[i] = basis_pnl
            total_basis += basis_pnl

        if s == 0 and holding:
            cost = eq * exit_cost_f
            eq -= cost
            total_costs += cost
            holding = False

        equity[i] = eq
        in_pos[i] = 1 if holding else 0

    eq_series = pd.Series(equity, index=times, name="equity")
    pos_series = pd.Series(in_pos, index=times, name="in_position")
    fund_series = pd.Series(fund_series, index=times, name="funding_pnl")
    basis_series = pd.Series(basis_series, index=times, name="basis_pnl")

    metrics = _compute_metrics(eq_series, periods_per_year, n_entries)

    return CarryV2Result(
        equity=eq_series,
        in_position=pos_series,
        funding_received=fund_series,
        basis_pnl=basis_series,
        n_entries=n_entries,
        total_funding=total_funding,
        total_basis=total_basis,
        total_costs=total_costs,
        initial_equity=initial_equity,
        final_equity=float(eq),
        metrics=metrics,
    )

"""Simulated fill logic — cash + units representation.

State:
  - cash: USD available
  - units: asset units held (0 if flat)
  - equity = cash + units * current_price

Target position is a fraction in [0, 1] representing desired exposure.
At execution price P: desired_units = (target * equity) / P
"""

from __future__ import annotations

from dataclasses import dataclass

from loguru import logger


@dataclass
class FillOutcome:
    cash: float
    units: float
    entry_price: float | None
    entry_time: str | None
    entry_equity: float | None
    trade_pnl: float | None
    trade_won: bool | None


def simulate_fill(
    *,
    cash: float,
    units: float,
    equity: float,
    entry_price: float | None,
    entry_time: str | None,
    entry_equity: float | None,
    target_position: float,
    fill_price: float,
    fill_time: str,
    cost_bps_per_side: float,
) -> FillOutcome:
    """Execute to reach target_position. Returns new cash/units state."""
    cf = cost_bps_per_side / 10_000.0
    target_position = max(0.0, min(1.0, float(target_position)))

    current_position_value = units * fill_price
    current_position_pct = (current_position_value / equity) if equity > 0 else 0.0

    if abs(current_position_pct - target_position) < 1e-9:
        return FillOutcome(cash, units, entry_price, entry_time, entry_equity, None, None)

    target_value = target_position * equity

    if target_value > current_position_value:
        # BUY the difference
        buy_value = target_value - current_position_value
        cost = buy_value * cf
        cash -= (buy_value + cost)
        units += buy_value / fill_price

        if current_position_pct < 1e-9:
            # Opening from flat
            entry_price = fill_price
            entry_time = fill_time
            entry_equity = equity
            logger.info("BUY {:.0%} {} @ {:.4f} cost={:.4f}",
                        target_position, fill_time, fill_price, cost)

        return FillOutcome(cash, units, entry_price, entry_time, entry_equity, None, None)

    else:
        # SELL the difference
        sell_value = current_position_value - target_value
        sell_units = sell_value / fill_price
        cost = sell_value * cf
        cash += (sell_value - cost)
        units -= sell_units

        if target_position < 1e-9:
            # Closing to flat
            trade_pnl = (sell_value - cost) - (units + sell_units) * 0 - 0
            # Simpler: entry equity was current equity when we opened.
            # When we opened, we deployed entry_equity * original_position.
            # Here, trade_pnl = total equity now - entry_equity_at_open.
            # Approximate: pnl = current equity after close - entry_equity
            new_equity = cash + units * fill_price
            trade_pnl = new_equity - (entry_equity if entry_equity is not None else new_equity)
            trade_won = trade_pnl > 0
            logger.info("SELL {} @ {:.4f} cost={:.4f} pnl={:.4f}",
                        fill_time, fill_price, cost, trade_pnl)
            return FillOutcome(
                cash=cash, units=0.0,
                entry_price=None, entry_time=None, entry_equity=None,
                trade_pnl=trade_pnl, trade_won=trade_won,
            )

        return FillOutcome(cash, units, entry_price, entry_time, entry_equity, None, None)


def mark_to_market(cash: float, units: float, close_price: float) -> float:
    return cash + units * close_price

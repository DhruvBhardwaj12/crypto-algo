"""Paper trading fill simulator — cash + units model."""

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
    cf = cost_bps_per_side / 10_000.0
    target_position = max(0.0, min(1.0, float(target_position)))

    current_value = units * fill_price
    target_value = target_position * equity

    if abs(current_value - target_value) < 1e-9:
        return FillOutcome(cash, units, entry_price, entry_time, entry_equity, None, None)

    if target_value > current_value:
        # BUY
        buy_value = target_value - current_value
        cost = buy_value * cf
        cash -= (buy_value + cost)
        units += buy_value / fill_price

        was_flat = (units - buy_value / fill_price) < 1e-12
        if was_flat:
            entry_price = fill_price
            entry_time = fill_time
            entry_equity = equity
            logger.info("BUY {:.0%} {} @ {:.4f} cost={:.4f}",
                        target_position, fill_time, fill_price, cost)
        return FillOutcome(cash, units, entry_price, entry_time, entry_equity, None, None)

    # SELL
    sell_value = current_value - target_value
    cost = sell_value * cf
    units -= sell_value / fill_price
    cash += (sell_value - cost)

    if target_position < 1e-9:
        new_equity = cash
        pnl = new_equity - (entry_equity if entry_equity is not None else new_equity)
        won = pnl > 0
        logger.info("SELL {} @ {:.4f} cost={:.4f} pnl={:.4f}",
                    fill_time, fill_price, cost, pnl)
        return FillOutcome(
            cash=cash, units=0.0,
            entry_price=None, entry_time=None, entry_equity=None,
            trade_pnl=pnl, trade_won=won,
        )

    return FillOutcome(cash, units, entry_price, entry_time, entry_equity, None, None)


def mark_to_market(cash: float, units: float, close_price: float) -> float:
    return cash + units * close_price

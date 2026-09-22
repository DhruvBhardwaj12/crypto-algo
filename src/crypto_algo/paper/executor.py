"""Simulated fill logic. Long-only, position size = 100% of equity."""

from __future__ import annotations

from dataclasses import dataclass

from loguru import logger


@dataclass
class FillOutcome:
    equity: float
    position: int
    units: float
    entry_price: float | None
    entry_time: str | None
    entry_equity: float | None
    trade_pnl: float | None
    trade_won: bool | None


def simulate_fill(
    *,
    equity: float,
    position: int,
    units: float,
    entry_price: float | None,
    entry_time: str | None,
    entry_equity: float | None,
    target_position: int,
    fill_price: float,
    fill_time: str,
    cost_bps_per_side: float,
) -> FillOutcome:
    """Execute target_position at fill_price. Update equity/units/position."""
    cf = cost_bps_per_side / 10_000.0

    if position == target_position:
        return FillOutcome(equity, position, units, entry_price, entry_time, entry_equity, None, None)

    if position == 0 and target_position == 1:
        # BUY
        cost = equity * cf
        investable = equity - cost
        new_units = investable / fill_price
        logger.info("BUY  {} @ {:.2f}  cost={:.2f}  units={:.6f}", fill_time, fill_price, cost, new_units)
        return FillOutcome(
            equity=investable,
            position=1,
            units=new_units,
            entry_price=fill_price,
            entry_time=fill_time,
            entry_equity=equity,
            trade_pnl=None,
            trade_won=None,
        )

    if position == 1 and target_position == 0:
        # SELL
        gross = units * fill_price
        cost = gross * cf
        final_cash = gross - cost
        trade_pnl = final_cash - (entry_equity if entry_equity is not None else equity)
        trade_won = trade_pnl > 0
        logger.info(
            "SELL {} @ {:.2f}  gross={:.2f}  cost={:.2f}  pnl={:.2f}",
            fill_time, fill_price, gross, cost, trade_pnl,
        )
        return FillOutcome(
            equity=final_cash,
            position=0,
            units=0.0,
            entry_price=None,
            entry_time=None,
            entry_equity=None,
            trade_pnl=trade_pnl,
            trade_won=trade_won,
        )

    raise ValueError(f"Unsupported transition {position} -> {target_position}")


def mark_to_market(units: float, position: int, equity_when_flat: float, close_price: float) -> float:
    """Current MTM equity."""
    if position == 0:
        return equity_when_flat
    return units * close_price

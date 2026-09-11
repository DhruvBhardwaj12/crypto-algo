"""Transaction cost model for an Indian Binance spot trader.

BUY:  0.100% exchange + 0.018% GST + 0.007% slippage = 0.125%  = 12.5 bps
SELL: 0.100% exchange + 0.018% GST + 1.000% TDS + 0.007% slippage = 1.125% = 112.5 bps
ROUND TRIP: 1.25%
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CostModel:
    """Buy/sell transaction costs in basis points (1 bp = 0.01%)."""

    buy_cost_bps: float = 12.5
    sell_cost_bps: float = 112.5

    @property
    def round_trip_bps(self) -> float:
        return self.buy_cost_bps + self.sell_cost_bps

    def cost_fraction(self, side: str) -> float:
        if side == "buy":
            return self.buy_cost_bps / 10_000.0
        if side == "sell":
            return self.sell_cost_bps / 10_000.0
        raise ValueError(f"Unknown side: {side!r}")
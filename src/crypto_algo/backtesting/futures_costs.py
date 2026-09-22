"""Transaction costs for a delta-neutral funding carry trade on Binance.

Two legs:
  1. Spot long:  taker fee + 18% GST on the fee.
  2. Perp short: taker fee only. No TDS, no GST (derivatives are not VDA).

Round-trip cost is ~0.34% on notional. Compare to ~1.25% for a single-leg
spot trade. This is the structural advantage that makes carry viable for
an Indian trader under the current tax regime.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FuturesCostModel:
    spot_taker_bps: float = 10.0     # 0.10% Binance spot taker
    gst_rate: float = 0.18           # 18% GST on exchange fees
    futures_taker_bps: float = 5.0   # 0.05% Binance USDT-M futures taker

    @property
    def spot_effective_bps(self) -> float:
        """Spot fee plus GST."""
        return self.spot_taker_bps * (1.0 + self.gst_rate)

    @property
    def entry_cost_bps(self) -> float:
        """Buy spot + short perp at open."""
        return self.spot_effective_bps + self.futures_taker_bps

    @property
    def exit_cost_bps(self) -> float:
        """Sell spot + close perp."""
        return self.spot_effective_bps + self.futures_taker_bps

    @property
    def round_trip_bps(self) -> float:
        return self.entry_cost_bps + self.exit_cost_bps

    def entry_cost_fraction(self) -> float:
        return self.entry_cost_bps / 10_000.0

    def exit_cost_fraction(self) -> float:
        return self.exit_cost_bps / 10_000.0
    
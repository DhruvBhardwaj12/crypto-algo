"""Persistent state for the paper trader.

State is JSON on disk. Every update writes atomically (write to .tmp,
rename). If the process crashes, we lose nothing.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional


@dataclass
class PaperState:
    symbol: str
    strategy_name: str
    initial_equity: float

    equity: float
    position: int = 0              # 0 or 1
    units: float = 0.0             # asset units held when in position
    entry_price: Optional[float] = None
    entry_time: Optional[str] = None
    entry_equity: Optional[float] = None

    n_trades: int = 0
    n_wins: int = 0
    n_losses: int = 0

    last_bar_time: Optional[str] = None       # open_time of last processed bar
    pending_signal: Optional[int] = None      # signal to execute at next bar open
    pending_signal_bar: Optional[str] = None  # bar on which signal was computed

    @classmethod
    def load_or_init(
        cls,
        path: Path,
        symbol: str,
        strategy_name: str,
        initial_equity: float,
    ) -> "PaperState":
        if path.exists():
            with open(path) as f:
                data = json.load(f)
            return cls(**data)
        return cls(
            symbol=symbol,
            strategy_name=strategy_name,
            initial_equity=initial_equity,
            equity=initial_equity,
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        with open(tmp, "w") as f:
            json.dump(asdict(self), f, indent=2, default=str)
        os.replace(tmp, path)
        
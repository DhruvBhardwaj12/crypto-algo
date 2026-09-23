"""Persistent paper trading state — cash + units representation.

Cash and units are tracked separately. Equity = cash + units * price.
This model handles fractional positions (from the risk engine) cleanly.
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

    cash: float
    units: float = 0.0

    entry_price: Optional[float] = None
    entry_time: Optional[str] = None
    entry_equity: Optional[float] = None

    n_trades: int = 0
    n_wins: int = 0
    n_losses: int = 0
    n_risk_rejections: int = 0
    n_risk_reductions: int = 0

    last_bar_time: Optional[str] = None
    pending_signal: Optional[float] = None       # fractional target in [0, 1]
    pending_signal_bar: Optional[str] = None

    peak_equity: float = 0.0
    day_start_equity: float = 0.0
    last_day: Optional[str] = None                # "YYYY-MM-DD" UTC

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
            # Backfill any missing fields from older-format state files.
            if "cash" not in data and "equity" in data:
                # Migrate old format.
                data["cash"] = data["equity"]
                data.pop("equity", None)
                data.pop("position", None)
            data.setdefault("n_risk_rejections", 0)
            data.setdefault("n_risk_reductions", 0)
            data.setdefault("peak_equity", initial_equity)
            data.setdefault("day_start_equity", initial_equity)
            data.setdefault("last_day", None)
            return cls(**data)
        return cls(
            symbol=symbol,
            strategy_name=strategy_name,
            initial_equity=initial_equity,
            cash=initial_equity,
            peak_equity=initial_equity,
            day_start_equity=initial_equity,
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        with open(tmp, "w") as f:
            json.dump(asdict(self), f, indent=2, default=str)
        os.replace(tmp, path)

    def equity(self, current_price: float) -> float:
        return self.cash + self.units * current_price
    
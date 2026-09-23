"""Risk engine — the last line of defense before execution.

A strategy proposes a target portfolio (weights per symbol). The risk
engine decides:

  - ALLOW: the proposal is within all limits
  - REDUCE: the proposal must be scaled down to fit limits
  - REJECT: the proposal must not be executed at all

No strategy can bypass the risk engine. If a limit is breached, the
engine can also force-flatten positions and trigger the kill switch.

Design notes:
  - The engine is deterministic. Same inputs → same output.
  - Every decision is logged with a reason.
  - Config is loaded once and frozen.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import pandas as pd
import yaml
from loguru import logger


class RiskDecision(Enum):
    ALLOW = "allow"
    REDUCE = "reduce"
    REJECT = "reject"


@dataclass
class RiskReport:
    decision: RiskDecision
    approved_weights: pd.Series
    reasons: list[str] = field(default_factory=list)
    kill_switch_triggered: bool = False
    drawdown_stop_triggered: bool = False


class RiskEngine:
    """Portfolio-level risk limiter."""

    def __init__(self, config_path: Path):
        with open(config_path, "r", encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f)

        # Snapshot of historical peak equity, used for drawdown calc.
        self._peak_equity: float = 0.0
        self._initial_equity: float | None = None
        self._day_start_equity: float | None = None

    # ---- State update -------------------------------------------------

    def update_equity(self, current_equity: float) -> None:
        """Call once per bar to update peak/daily tracking."""
        if self._initial_equity is None:
            self._initial_equity = current_equity
            self._day_start_equity = current_equity
        self._peak_equity = max(self._peak_equity, current_equity)

    def new_day(self, current_equity: float) -> None:
        """Call at start of each UTC day."""
        self._day_start_equity = current_equity

    # ---- Checks -------------------------------------------------------

    def _check_kill_switch(self) -> tuple[bool, str | None]:
        if self.cfg["operational"]["kill_switch"]:
            return True, "config kill_switch is True"
        path = Path(self.cfg["operational"]["kill_switch_file"])
        if path.exists():
            return True, f"kill switch file exists at {path}"
        return False, None

    def _check_total_loss(self, equity: float) -> tuple[bool, str | None]:
        if self._initial_equity is None or self._initial_equity <= 0:
            return False, None
        loss = (equity - self._initial_equity) / self._initial_equity
        limit = -self.cfg["portfolio"]["max_total_loss_pct"]
        if loss < limit:
            return True, f"total loss {loss:.2%} exceeds {limit:.2%}"
        return False, None

    def _check_drawdown(self, equity: float) -> tuple[bool, str | None]:
        if self._peak_equity <= 0:
            return False, None
        dd = (equity - self._peak_equity) / self._peak_equity
        limit = -self.cfg["portfolio"]["max_drawdown_pct"]
        if dd < limit:
            return True, f"drawdown {dd:.2%} exceeds {limit:.2%}"
        return False, None

    def _check_daily_loss(self, equity: float) -> tuple[bool, str | None]:
        if self._day_start_equity is None or self._day_start_equity <= 0:
            return False, None
        loss = (equity - self._day_start_equity) / self._day_start_equity
        limit = -self.cfg["portfolio"]["max_daily_loss_pct"]
        if loss < limit:
            return True, f"daily loss {loss:.2%} exceeds {limit:.2%}"
        return False, None

    def _scale_to_limits(self, weights: pd.Series, equity: float) -> tuple[pd.Series, list[str]]:
        """Apply position and exposure limits to a proposed weight vector."""
        reasons: list[str] = []
        max_pos = self.cfg["position"]["max_position_pct"]
        max_gross = self.cfg["position"]["max_gross_exposure"]
        max_net = self.cfg["position"]["max_net_exposure"]

        scaled = weights.copy()

        # Per-position cap.
        breach = scaled.abs() > max_pos
        if breach.any():
            reasons.append(f"scaled {int(breach.sum())} position(s) to cap {max_pos}")
            scaled[breach] = scaled[breach].clip(-max_pos, max_pos)

        # Gross exposure cap.
        gross = scaled.abs().sum()
        if gross > max_gross:
            scale = max_gross / gross
            reasons.append(f"gross {gross:.3f} → {max_gross}: multiplied by {scale:.3f}")
            scaled = scaled * scale

        # Net exposure cap.
        net = scaled.sum()
        if abs(net) > max_net:
            scale = max_net / abs(net)
            reasons.append(f"net {net:+.3f} → {max_net:.3f}: multiplied by {scale:.3f}")
            scaled = scaled * scale

        return scaled, reasons

    # ---- Main entry point ---------------------------------------------

    def evaluate(
        self,
        current_equity: float,
        current_weights: pd.Series,
        proposed_weights: pd.Series,
    ) -> RiskReport:
        """Evaluate a proposed target portfolio.

        Parameters
        ----------
        current_equity: mark-to-market portfolio value.
        current_weights: current portfolio weights.
        proposed_weights: what the strategy wants to hold.
        """
        self.update_equity(current_equity)

        reasons: list[str] = []

        # 1. Kill switch.
        kill, kill_reason = self._check_kill_switch()
        if kill:
            logger.warning("KILL SWITCH: {}", kill_reason)
            return RiskReport(
                decision=RiskDecision.REJECT,
                approved_weights=pd.Series(0.0, index=proposed_weights.index),
                reasons=[f"kill switch: {kill_reason}"],
                kill_switch_triggered=True,
            )

        # 2. Total loss.
        breach, reason = self._check_total_loss(current_equity)
        if breach:
            reasons.append(reason)
            return RiskReport(
                decision=RiskDecision.REJECT,
                approved_weights=pd.Series(0.0, index=proposed_weights.index),
                reasons=reasons,
            )

        # 3. Drawdown.
        breach, reason = self._check_drawdown(current_equity)
        if breach:
            reasons.append(reason)
            return RiskReport(
                decision=RiskDecision.REJECT,
                approved_weights=pd.Series(0.0, index=proposed_weights.index),
                reasons=reasons,
                drawdown_stop_triggered=True,
            )

        # 4. Daily loss.
        breach, reason = self._check_daily_loss(current_equity)
        if breach:
            reasons.append(reason)
            return RiskReport(
                decision=RiskDecision.REJECT,
                approved_weights=pd.Series(0.0, index=proposed_weights.index),
                reasons=reasons,
            )

        # 5. Position / exposure scaling.
        aligned_proposed = proposed_weights.reindex(current_weights.index).fillna(0.0)
        scaled, scale_reasons = self._scale_to_limits(aligned_proposed, current_equity)

        if scale_reasons:
            reasons.extend(scale_reasons)
            return RiskReport(
                decision=RiskDecision.REDUCE,
                approved_weights=scaled,
                reasons=reasons,
            )

        return RiskReport(
            decision=RiskDecision.ALLOW,
            approved_weights=aligned_proposed,
            reasons=["all limits satisfied"],
        )
    
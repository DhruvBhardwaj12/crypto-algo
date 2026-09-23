"""Test the risk engine's decisions.

We simulate several scenarios and confirm the engine behaves correctly.

Run:
    uv run python scripts/test_risk_engine.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from crypto_algo.risk.engine import RiskDecision, RiskEngine

CONFIG = Path("config/risk_limits.yaml")


def _report(name: str, report) -> None:
    print(f"\n--- {name} ---")
    print(f"  decision:   {report.decision.value}")
    print(f"  reasons:    {report.reasons}")
    print(f"  approved:   {report.approved_weights.to_dict()}")


def main() -> None:
    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"]

        # Scenario 1 — all limits satisfied.
    # 4 × 0.12 = 0.48 net, which is below the 0.50 cap.
    engine = RiskEngine(CONFIG)
    cur = pd.Series(0.0, index=symbols)
    prop = pd.Series([0.12, 0.12, 0.12, 0.12], index=symbols)
    r = engine.evaluate(10_000, cur, prop)
    _report("S1: small positions, within all limits", r)
    assert r.decision == RiskDecision.ALLOW, "should be ALLOW"

    # Scenario 2 — single position exceeds 20% cap.
    engine = RiskEngine(CONFIG)
    prop = pd.Series([0.40, 0.10, 0.10, 0.10], index=symbols)
    r = engine.evaluate(10_000, cur, prop)
    _report("S2: single position 40% exceeds 20% cap", r)
    assert r.decision == RiskDecision.REDUCE, "should be REDUCE"
    assert r.approved_weights.abs().max() <= 0.20 + 1e-9

    # Scenario 3 — gross exposure exceeds 1.0.
    engine = RiskEngine(CONFIG)
    prop = pd.Series([0.20, 0.20, 0.20, 0.20], index=symbols)  # gross = 0.80
    prop_long = pd.Series([0.30, 0.30, 0.30, 0.30], index=symbols)  # gross = 1.20
    r = engine.evaluate(10_000, cur, prop_long)
    _report("S3: gross 1.20 exceeds 1.00 cap", r)
    assert r.decision == RiskDecision.REDUCE, "should be REDUCE"
    assert r.approved_weights.abs().sum() <= 1.00 + 1e-9

    # Scenario 4 — daily loss exceeds 3%.
    engine = RiskEngine(CONFIG)
    engine.update_equity(10_000)  # initial
    engine.new_day(10_000)         # day starts at 10k
    r = engine.evaluate(9_600, cur, prop)  # lost 4%
    _report("S4: daily loss -4% exceeds -3% cap", r)
    assert r.decision == RiskDecision.REJECT, "should be REJECT"

    # Scenario 5 — drawdown exceeds 15%.
    engine = RiskEngine(CONFIG)
    engine.update_equity(10_000)  # peak
    r = engine.evaluate(8_400, cur, prop)  # 16% DD
    _report("S5: drawdown -16% exceeds -15% cap", r)
    assert r.decision == RiskDecision.REJECT, "should be REJECT"

    # Scenario 6 — total loss exceeds 25%.
    engine = RiskEngine(CONFIG)
    engine.update_equity(10_000)  # initial
    r = engine.evaluate(7_400, cur, prop)  # 26% loss from initial
    _report("S6: total loss -26% exceeds -25% cap", r)
    assert r.decision == RiskDecision.REJECT, "should be REJECT"

    # Scenario 7 — kill switch file present.
    kill_file = Path("research/KILL_SWITCH")
    kill_file.parent.mkdir(parents=True, exist_ok=True)
    kill_file.touch()
    engine = RiskEngine(CONFIG)
    r = engine.evaluate(10_000, cur, prop)
    _report("S7: kill switch file present", r)
    assert r.kill_switch_triggered, "should trigger kill switch"
    kill_file.unlink()  # clean up

    print("\nAll risk engine tests passed.")


if __name__ == "__main__":
    main()

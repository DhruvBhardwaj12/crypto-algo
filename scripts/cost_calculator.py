"""Cost vs. edge reality calculator.

Run:
    uv run python scripts/cost_calculator.py
"""

def calculate(
    trades_per_day: float,
    edge_bps_per_trade: float,
    cost_bps_round_trip: float,
    days_per_year: int = 250,
) -> dict:
    trades_per_year = trades_per_day * days_per_year
    net_bps_per_trade = edge_bps_per_trade - cost_bps_round_trip
    annual_net_bps = net_bps_per_trade * trades_per_year
    annual_net_pct = annual_net_bps / 100.0
    return {
        "trades_per_year": trades_per_year,
        "net_bps_per_trade": net_bps_per_trade,
        "annual_net_pct": annual_net_pct,
        "verdict": "PROFITABLE" if annual_net_pct > 0 else "LOSS",
    }


def main() -> None:
    print("Cost vs. Edge Reality Calculator")
    print("=" * 70)
    print()
    scenarios = [
        # (name, trades/day, edge bps, cost bps round trip)
        ("4H Donchian (actual EXP-004)", 0.2, 40, 10),
        ("4H TSMOM (actual LAB-001)", 0.3, 30, 10),
        ("1H trend strategy", 1.0, 15, 10),
        ("15m mean reversion", 2.0, 4, 10),
        ("15m mean reversion (maker fees)", 2.0, 4, 5),
        ("5m RSI(2) (actual LAB-002)", 13.0, 1.7, 10),
        ("1m high-frequency", 50.0, 0.5, 10),
        ("1m with all optimizations", 50.0, 3.0, 4),
    ]
    print(f"{'Scenario':38s} {'Trades/yr':>10s} {'Net bps/tr':>10s} {'Annual %':>10s}  Verdict")
    print("-" * 90)
    for name, tpd, edge, cost in scenarios:
        r = calculate(tpd, edge, cost)
        print(
            f"{name:38s} {r['trades_per_year']:>10.0f} "
            f"{r['net_bps_per_trade']:>10.2f} {r['annual_net_pct']:>9.2f}%  {r['verdict']}"
        )
    print()
    print("Notes:")
    print("  - edge_bps is the GROSS edge before cost (per trade, in bps).")
    print("  - cost_bps is round-trip cost (entry + exit combined).")
    print("  - A strategy is profitable iff edge > cost, INDEPENDENT of frequency.")
    print("  - But annual_return = (edge - cost) * trades_per_year, so frequency")
    print("    amplifies whatever the per-trade relationship is.")


if __name__ == "__main__":
    main()
    
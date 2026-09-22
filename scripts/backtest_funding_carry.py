"""Backtest funding carry on BTCUSDT and ETHUSDT.

Run:
    uv run python scripts/backtest_funding_carry.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from loguru import logger

from crypto_algo.backtesting.carry_simulator import run_carry
from crypto_algo.backtesting.futures_costs import FuturesCostModel
from crypto_algo.strategies.funding_carry import funding_carry_signal

START = "2020-01-01"
END = "2025-01-01"
SYMBOLS = ["BTCUSDT", "ETHUSDT"]
INITIAL_EQUITY = 10_000.0
COST = FuturesCostModel()


def _load_funding(symbol: str) -> pd.DataFrame:
    path = Path(f"data/raw/binance/funding/{symbol}/{START}_{END}.parquet")
    df = pd.read_parquet(path)
    df = df.sort_values("funding_time").reset_index(drop=True)
    return df


def _annualized_rate(rate_per_event: float) -> float:
    """3 funding events per day × 365 days."""
    return (1.0 + rate_per_event) ** (3 * 365) - 1.0


def main() -> None:
    # --- A quick look at the funding distribution ---
    print("\n=== Funding rate distribution (per 8h event) ===\n")
    for symbol in SYMBOLS:
        f = _load_funding(symbol)
        r = f["funding_rate"]
        mean_ann = _annualized_rate(float(r.mean()))
        print(f"{symbol}:")
        print(f"  N records:        {len(f)}")
        print(f"  range:            {f['funding_time'].iloc[0].date()} -> {f['funding_time'].iloc[-1].date()}")
        print(f"  mean per 8h:      {r.mean():+.6f}  ({mean_ann * 100:+.2f}% annualized)")
        print(f"  median per 8h:    {r.median():+.6f}")
        print(f"  std:              {r.std():.6f}")
        print(f"  % events > 0:     {(r > 0).mean():.1%}")
        print(f"  % events > 0.5bp: {(r > 0.00005).mean():.1%}")
        print(f"  min:              {r.min():+.6f}")
        print(f"  max:              {r.max():+.6f}")
        print()

    # --- Run the strategy on each symbol ---
    results = {}
    for symbol in SYMBOLS:
        funding = _load_funding(symbol)
        signal = funding_carry_signal(funding["funding_rate"])
        signal.index = funding.index

        result = run_carry(
            funding=funding,
            signal=signal,
            cost_model=COST,
            initial_equity=INITIAL_EQUITY,
        )
        results[symbol] = result

        print(f"\n=== {symbol} — funding carry ===")
        print(f"Final equity: {result.final_equity:,.2f} (start {INITIAL_EQUITY:,.2f})")
        print(f"Entries: {result.n_entries}  Exits: {result.n_exits}")
        print(f"Cumulative funding income: {result.funding_income:,.2f}")
        print(f"Total costs paid:          {result.total_costs:,.2f}")
        print("Metrics:")
        for k, v in result.metrics.items():
            if isinstance(v, float):
                print(f"  {k:16s} {v:,.4f}")
            else:
                print(f"  {k:16s} {v}")

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(12, 6))
    for symbol, res in results.items():
        ax.plot(res.equity.index, res.equity.values, label=f"{symbol} carry")
    ax.set_title(f"Funding carry on BTC/ETH — {START} to {END}")
    ax.set_xlabel("Time (UTC)")
    ax.set_ylabel("Equity (USD)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = Path("research/backtest_funding_carry.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=120)
    logger.success("Saved plot to {}", out)

    print()
    print(f"Cost model (round trip): {COST.round_trip_bps:.2f} bps "
          f"= {COST.round_trip_bps / 100:.4f}%")


if __name__ == "__main__":
    main()
    
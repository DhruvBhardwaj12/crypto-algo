"""Run MA crossover backtest on DAILY data with Indian cost model.

EXP-002: same strategy, same parameters as EXP-001, only the timeframe
changed (1h -> 1d). Isolates whether EXP-001's failure was driven by
the timeframe (hourly whipsaws + high turnover) or by the strategy logic.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from loguru import logger

from crypto_algo.backtesting.costs import CostModel
from crypto_algo.backtesting.simulator import buy_and_hold_equity, run_backtest
from crypto_algo.data.loaders import load_parquet
from crypto_algo.strategies.ma_crossover import ma_crossover_target

FAST_WINDOW = 20
SLOW_WINDOW = 100
INITIAL_EQUITY = 10_000.0
COST = CostModel()
START = "2020-01-01"
END = "2025-01-01"


def _fmt(v: float) -> str:
    if v != v:
        return "n/a"
    if abs(v) < 10:
        return f"{v:.4f}"
    return f"{v:,.4f}"


def main() -> None:
    data_dir = Path("data/raw/binance")
    symbols = ["BTCUSDT", "ETHUSDT"]

    results = {}
    for symbol in symbols:
        path = data_dir / symbol / "1d" / f"{START}_{END}.parquet"
        df = load_parquet(path)

        target = ma_crossover_target(
            df["close"], fast_window=FAST_WINDOW, slow_window=SLOW_WINDOW
        ).reset_index(drop=True)
        target.index = df.index

        result = run_backtest(
            df=df,
            target_position=target,
            cost_model=COST,
            initial_equity=INITIAL_EQUITY,
            periods_per_year=365,
        )
        results[symbol] = result

        bh = buy_and_hold_equity(df, INITIAL_EQUITY)
        bh_total_return = float(bh.iloc[-1] / bh.iloc[0] - 1.0)

        print(f"\n=== {symbol} — MA({FAST_WINDOW}/{SLOW_WINDOW}) daily ===")
        print(f"Rows: {len(df)}  ({df['open_time'].iloc[0]} -> {df['open_time'].iloc[-1]})")
        print(f"Final equity: {result.final_equity:,.2f} (start {INITIAL_EQUITY:,.2f})")
        print(f"Buy & hold final: {bh.iloc[-1]:,.2f}  (total return {bh_total_return:+.2%})")
        print(f"Trades executed: {len(result.trades)}")
        print("Metrics:")
        for k, v in result.metrics.items():
            if isinstance(v, float):
                print(f"  {k:16s} {_fmt(v)}")
            else:
                print(f"  {k:16s} {v}")

    fig, ax = plt.subplots(figsize=(12, 6))
    for symbol, res in results.items():
        ax.plot(res.equity.index, res.equity.values, label=f"{symbol} MA strategy")

    btc = load_parquet(data_dir / "BTCUSDT" / "1d" / f"{START}_{END}.parquet")
    bh = buy_and_hold_equity(btc, INITIAL_EQUITY)
    ax.plot(bh.index, bh.values, label="BTC buy & hold", linestyle="--", alpha=0.7)

    ax.set_title(f"Daily MA Crossover ({FAST_WINDOW}/{SLOW_WINDOW}) — {START} to {END}")
    ax.set_xlabel("Time (UTC)")
    ax.set_ylabel("Equity (USD)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    out_path = Path("research/backtest_ma_daily.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=120)
    logger.success("Saved plot to {}", out_path)

    print()
    print("Cost model:")
    print(f"  buy:        {COST.buy_cost_bps:.1f} bps ({COST.buy_cost_bps/100:.4f}%)")
    print(f"  sell:       {COST.sell_cost_bps:.1f} bps ({COST.sell_cost_bps/100:.4f}%)")
    print(f"  round trip: {COST.round_trip_bps:.1f} bps ({COST.round_trip_bps/100:.4f}%)")


if __name__ == "__main__":
    main()
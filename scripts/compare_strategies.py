"""Compare EXP-004 (Donchian+RSI) and TSMOM (LAB-001) return streams.

Runs both strategies on the same data, computes their daily return
correlation, and produces a combined-portfolio backtest (equal weight).

Run:
    uv run python scripts/compare_strategies.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from crypto_algo.backtesting.simulator import run_backtest
from crypto_algo.data.loaders import load_parquet
from crypto_algo.strategies.donchian_rsi import donchian_rsi_target, tag_with_4h_regime
from crypto_algo.strategies.tsmom import tsmom_target

START = "2020-01-01"
END = "2025-01-01"
SYMBOLS = ["BTCUSDT", "ETHUSDT"]
INITIAL_EQUITY = 10_000.0
PERIODS_PER_YEAR_4H = 6 * 365
FUTURES_TAKER_BPS = 5.0

# Per-symbol chosen params from validation
DONCHIAN_PARAMS = {"dc": 55, "dc_exit": 20, "rsi4h": 52.0}
TSMOM_LOOKBACK = {"BTCUSDT": 168, "ETHUSDT": 84}


class _Cost:
    def __init__(self, bps: float):
        self.cost_bps = bps

    @property
    def buy_cost_bps(self) -> float:
        return self.cost_bps

    @property
    def sell_cost_bps(self) -> float:
        return self.cost_bps

    @property
    def round_trip_bps(self) -> float:
        return 2 * self.cost_bps

    def cost_fraction(self, side: str) -> float:
        return self.cost_bps / 10_000.0


COST = _Cost(FUTURES_TAKER_BPS)


def _load_4h(symbol: str) -> pd.DataFrame:
    return load_parquet(
        Path(f"data/raw/binance/futures/{symbol}/4h/{START}_{END}.parquet")
    )


def _load_1h(symbol: str) -> pd.DataFrame:
    return load_parquet(
        Path(f"data/raw/binance/futures/{symbol}/1h/{START}_{END}.parquet")
    )


def main() -> None:
    for symbol in SYMBOLS:
        print(f"\n{'=' * 72}")
        print(f"  {symbol}  —  EXP-004 vs TSMOM vs combined")
        print(f"{'=' * 72}")

        df_1h = _load_1h(symbol)
        df_4h = _load_4h(symbol)

        # --- Strategy A: Donchian + RSI on 1H ---
        rsi_4h = tag_with_4h_regime(df_1h, df_4h, rsi_period=14)
        target_a = donchian_rsi_target(df_1h, rsi_4h, **{
            "donchian_entry_lookback": DONCHIAN_PARAMS["dc"],
            "donchian_exit_lookback": DONCHIAN_PARAMS["dc_exit"],
            "rsi_1h_period": 14,
            "rsi_1h_entry": 50.0,
            "rsi_4h_entry": DONCHIAN_PARAMS["rsi4h"],
            "rsi_4h_exit": 48.0,
        }).reset_index(drop=True)
        target_a.index = df_1h.index

        result_a = run_backtest(df_1h, target_a, COST, INITIAL_EQUITY, PERIODS_PER_YEAR_4H)

        # --- Strategy B: TSMOM on 4H ---
        lb = TSMOM_LOOKBACK[symbol]
        target_b = tsmom_target(df_4h["close"], lookback_n=lb).reset_index(drop=True)
        target_b.index = df_4h.index

        result_b = run_backtest(df_4h, target_b, COST, INITIAL_EQUITY, PERIODS_PER_YEAR_4H)

        # --- Daily returns for correlation ---
        rets_a = result_a.returns.resample("1D").apply(lambda r: (1 + r).prod() - 1).dropna()
        rets_b = result_b.returns.resample("1D").apply(lambda r: (1 + r).prod() - 1).dropna()

        df_ret = pd.concat({"donchian": rets_a, "tsmom": rets_b}, axis=1).dropna()
        corr = df_ret.corr().iloc[0, 1]

        print(f"\nDaily return correlation (Donchian vs TSMOM): {corr:+.3f}")
        print(f"Days of overlap: {len(df_ret)}")

        # --- Combined portfolio: 50/50 daily rebalance ---
        # Simply average daily returns as a first-order approximation.
        combined_ret = 0.5 * df_ret["donchian"] + 0.5 * df_ret["tsmom"]
        combined_equity = INITIAL_EQUITY * (1 + combined_ret).cumprod()

        def _stats(returns: pd.Series) -> dict:
            ann = np.sqrt(365)
            mean = returns.mean()
            std = returns.std()
            sharpe = mean / std * ann if std > 0 else float("nan")
            eq = (1 + returns).cumprod()
            peak = eq.cummax()
            dd = (eq / peak - 1).min()
            total = eq.iloc[-1] - 1
            cagr = (1 + total) ** (365 / len(returns)) - 1 if len(returns) > 0 else float("nan")
            return {
                "sharpe": sharpe,
                "cagr": cagr,
                "total": total,
                "max_dd": dd,
            }

        stats_a = _stats(df_ret["donchian"])
        stats_b = _stats(df_ret["tsmom"])
        stats_c = _stats(combined_ret)

        print(f"\n{'':20s} {'Donchian':>12s} {'TSMOM':>12s} {'Combined':>12s}")
        for k in ["total", "cagr", "sharpe", "max_dd"]:
            print(f"  {k:18s} {stats_a[k]:>12.4f} {stats_b[k]:>12.4f} {stats_c[k]:>12.4f}")

        combined_equity.to_csv(f"research/combined_equity_{symbol}.csv")


if __name__ == "__main__":
    main()
    
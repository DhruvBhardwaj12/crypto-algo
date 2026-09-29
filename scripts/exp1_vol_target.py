"""Experiment 1 — Volatility-targeted position sizing.

Compare base TSMOM (fixed position) against vol-targeted TSMOM
(position scaled inversely to realized vol).

Run:
    uv run python scripts/exp1_vol_target.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from crypto_algo.backtesting.simulator import run_backtest
from crypto_algo.data.loaders import load_parquet
from crypto_algo.strategies.tsmom import tsmom_target

INITIAL_EQUITY = 10_000.0
PERIODS_PER_YEAR = 6 * 365
COST_BPS = 5.0

# Frozen parameters
LOOKBACK_BTC = 168
LOOKBACK_ETH = 84
TARGET_VOL = 0.02       # 2% per 4H bar ~ 22% daily-adjusted
VOL_WINDOW = 30         # 5 days of 4H bars
MAX_POSITION = 1.0

WINDOWS = [
    ("HISTORICAL 2020-2025", "2020-01-01", "2025-01-01"),
    ("RECENT 2025-2026",     "2025-01-01", "2026-09-23"),
]


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


def vol_target_signal(
    close: pd.Series,
    base_signal: pd.Series,
    vol_window: int = VOL_WINDOW,
    median_window: int = 200,
    max_pos: float = MAX_POSITION,
) -> pd.Series:
    """Scale base_signal by min(1, trailing_median_vol / realized_vol)."""
    log_ret = np.log(close).diff()
    realized_vol = log_ret.rolling(vol_window, min_periods=vol_window).std()
    median_vol = realized_vol.rolling(median_window, min_periods=vol_window).median()
    scale = (median_vol / realized_vol).clip(upper=max_pos)
    scale = scale.fillna(0.0)
    scaled = (base_signal * scale).clip(0.0, max_pos)
    return scaled


def _stats(equity: pd.Series) -> dict:
    rets = equity.pct_change().dropna()
    total = float(equity.iloc[-1] / equity.iloc[0] - 1.0)
    years = len(equity) / PERIODS_PER_YEAR
    cagr = float((equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1) if years > 0 and equity.iloc[-1] > 0 else float("nan")
    std = float(rets.std())
    sharpe = float(rets.mean() / std * np.sqrt(PERIODS_PER_YEAR)) if std > 0 else float("nan")
    peak = equity.cummax()
    dd = float((equity / peak - 1.0).min())
    calmar = cagr / abs(dd) if dd < 0 else float("nan")
    return {"total_return": total, "cagr": cagr, "sharpe": sharpe, "max_dd": dd, "calmar": calmar}


def _fmt(v: float) -> str:
    if v != v:
        return "n/a"
    return f"{v:+.3f}"


def main() -> None:
    for symbol in ["BTCUSDT", "ETHUSDT"]:
        lookback = LOOKBACK_BTC if symbol == "BTCUSDT" else LOOKBACK_ETH

        for label, start, end in WINDOWS:
            path = Path(f"data/raw/binance/futures/{symbol}/4h/{start}_{end}.parquet")
            if not path.exists():
                print(f"\n[{symbol} {label}] no data")
                continue

            df = load_parquet(path)
            base = tsmom_target(df["close"], lookback_n=lookback).reset_index(drop=True)
            base.index = df.index

            vol_sig = vol_target_signal(df["close"], base).reset_index(drop=True)
            vol_sig.index = df.index

            r_base = run_backtest(df, base, _Cost(COST_BPS), INITIAL_EQUITY, PERIODS_PER_YEAR)
            r_vol = run_backtest(df, vol_sig, _Cost(COST_BPS), INITIAL_EQUITY, PERIODS_PER_YEAR)

            s_base = _stats(r_base.equity)
            s_vol = _stats(r_vol.equity)

            print(f"\n{'=' * 78}")
            print(f"  {symbol}  —  {label}")
            print(f"{'=' * 78}")
            print(f"  {'metric':>14s}  {'base':>10s}  {'vol-target':>12s}  {'delta':>10s}")
            for key in ["total_return", "cagr", "sharpe", "max_dd", "calmar"]:
                delta = s_vol[key] - s_base[key]
                print(f"  {key:>14s}  {_fmt(s_base[key]):>10s}  {_fmt(s_vol[key]):>12s}  {_fmt(delta):>10s}")

            # Exposure summary
            avg_exposure_base = float(base.mean())
            avg_exposure_vol = float(vol_sig.mean())
            print(f"  {'avg_exposure':>14s}  {avg_exposure_base:>10.3f}  {avg_exposure_vol:>12.3f}  {avg_exposure_vol - avg_exposure_base:>+10.3f}")


if __name__ == "__main__":
    main()

"""Backtest funding-rate contrarian strategy.

Hypothesis (from signal scan): extreme funding z-scores predict negative
forward returns (crowded positioning reverses). Test as long-only
contrarian: go long when funding is very negative (shorts crowded).

Rule:
  z = 30-event rolling z-score of funding rate
  Enter long when z < -1.5
  Exit to cash when z > -0.5
  Otherwise hold current state
  Execute at next 4H bar's open, long-only, unlevered.

Run:
    uv run python scripts/backtest_funding_z.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from crypto_algo.backtesting.simulator import run_backtest
from crypto_algo.data.loaders import load_parquet

INITIAL_EQUITY = 10_000.0
PERIODS_PER_YEAR = 6 * 365
COST_BPS = 5.0

# Frozen parameters
Z_WINDOW = 30
ENTRY_Z = -1.5
EXIT_Z = -0.5

SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
    "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "LINKUSDT", "DOTUSDT",
    "LTCUSDT", "ATOMUSDT", "NEARUSDT", "FILUSDT", "ARBUSDT",
    "OPUSDT", "INJUSDT", "SUIUSDT", "TRXUSDT",
]

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


def load_inputs(symbol: str, start: str, end: str):
    perp_path = Path(f"data/raw/binance/futures/{symbol}/4h/{start}_{end}.parquet")
    fund_path = Path(f"data/raw/binance/funding/{symbol}/{start}_{end}.parquet")
    if not perp_path.exists() or not fund_path.exists():
        return None, None
    df = load_parquet(perp_path).reset_index(drop=True)
    funding = pd.read_parquet(fund_path).sort_values("funding_time").reset_index(drop=True)
    return df, funding


def funding_z_on_4h(df: pd.DataFrame, funding: pd.DataFrame) -> pd.Series:
    """For each 4H bar, return the funding z-score from the most recent CLOSED event.

    All computations use only funding events that occurred strictly before
    the 4H bar's close (merge_asof with allow_exact_matches=False).
    """
    f = funding[["funding_time", "funding_rate"]].copy()
    # Compute rolling z on the funding-event grid (causal — rolling uses past only).
    mean = f["funding_rate"].rolling(Z_WINDOW, min_periods=Z_WINDOW).mean()
    std = f["funding_rate"].rolling(Z_WINDOW, min_periods=Z_WINDOW).std()
    f["z"] = (f["funding_rate"] - mean) / std.replace(0, np.nan)

    # Align to 4H bars by close_time, strictly before.
    left = df[["close_time"]].reset_index(drop=True)
    merged = pd.merge_asof(
        left,
        f[["funding_time", "z"]],
        left_on="close_time",
        right_on="funding_time",
        direction="backward",
        allow_exact_matches=False,
    )
    return merged["z"].reset_index(drop=True)


def funding_z_signal(z_series: pd.Series) -> pd.Series:
    """State machine: 1 = long, 0 = cash. Shifted 1 bar for execution."""
    z = z_series.to_numpy()
    n = len(z)
    state = np.zeros(n, dtype=float)
    in_pos = False
    for i in range(n):
        zi = z[i]
        if np.isnan(zi):
            state[i] = 1.0 if in_pos else 0.0
            continue
        if not in_pos and zi < ENTRY_Z:
            in_pos = True
        elif in_pos and zi > EXIT_Z:
            in_pos = False
        state[i] = 1.0 if in_pos else 0.0
    sig = pd.Series(state)
    return sig.shift(1).fillna(0.0)


def _stats(equity: pd.Series) -> dict:
    rets = equity.pct_change().dropna()
    total = float(equity.iloc[-1] / equity.iloc[0] - 1.0)
    years = len(equity) / PERIODS_PER_YEAR
    cagr = float((equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1) if years > 0 and equity.iloc[-1] > 0 else float("nan")
    std = float(rets.std())
    sharpe = float(rets.mean() / std * np.sqrt(PERIODS_PER_YEAR)) if std > 0 else float("nan")
    peak = equity.cummax()
    dd = float((equity / peak - 1.0).min())
    return {"total_return": total, "cagr": cagr, "sharpe": sharpe, "max_dd": dd}


def _fmt(v: float) -> str:
    if v != v:
        return "n/a"
    return f"{v:+.3f}" if abs(v) < 10 else f"{v:+,.2f}"


def main() -> None:
    for label, start, end in WINDOWS:
        print(f"\n{'=' * 88}")
        print(f"  Funding Z-Score Contrarian (long-only) — {label}")
        print(f"  Rule: enter if z < {ENTRY_Z}, exit if z > {EXIT_Z}, window={Z_WINDOW}")
        print(f"{'=' * 88}")

        per_symbol = []
        all_stats = []

        print(f"\n  {'symbol':>10s}  {'bars':>6s}  {'entries':>8s}  "
              f"{'total_ret':>10s}  {'sharpe':>8s}  {'max_dd':>9s}")

        for symbol in SYMBOLS:
            df, funding = load_inputs(symbol, start, end)
            if df is None or funding is None or len(df) < 500:
                continue

            z = funding_z_on_4h(df, funding)
            sig = funding_z_signal(z)
            sig.index = df.index

            result = run_backtest(df, sig, _Cost(COST_BPS), INITIAL_EQUITY, PERIODS_PER_YEAR)
            s = _stats(result.equity)
            all_stats.append(s)
            per_symbol.append(result.equity / result.equity.iloc[0])

            n_entries = int(((sig.diff() == 1.0).sum()))
            print(f"  {symbol:>10s}  {len(df):>6d}  {n_entries:>8d}  "
                  f"{_fmt(s['total_return']):>10s}  {_fmt(s['sharpe']):>8s}  "
                  f"{_fmt(s['max_dd']):>9s}")

        if not per_symbol:
            print("  No symbols loaded.")
            continue

        # Portfolio: equal-weight average of normalized equity curves.
        min_len = min(len(e) for e in per_symbol)
        norm = pd.concat([e.iloc[:min_len] for e in per_symbol], axis=1).mean(axis=1)
        port_eq = norm * INITIAL_EQUITY
        ps = _stats(port_eq)

        mean_sharpe = float(np.nanmean([s["sharpe"] for s in all_stats]))
        n_pos = sum(1 for s in all_stats if s["total_return"] > 0)

        print(f"\n  --- Portfolio (equal-weight {len(per_symbol)} symbols) ---")
        print(f"    Total return:      {_fmt(ps['total_return'])}")
        print(f"    CAGR:              {_fmt(ps['cagr'])}")
        print(f"    Sharpe:            {_fmt(ps['sharpe'])}")
        print(f"    Max drawdown:      {_fmt(ps['max_dd'])}")
        print(f"    Mean single Sharpe: {mean_sharpe:+.3f}")
        print(f"    Symbols profitable: {n_pos}/{len(all_stats)}")


if __name__ == "__main__":
    main()
    
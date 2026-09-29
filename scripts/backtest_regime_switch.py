"""Conditional regime switching strategy.

Uses rolling lag-1 autocorrelation of 4H log returns to decide which
strategy to run. If autocorr is positive -> momentum regime. If
negative -> reversion regime. If |autocorr| < threshold -> flat.

Frozen parameters (no tuning):
  AUTOCORR_WINDOW = 500        (~83 days of 4H)
  AUTOCORR_THRESHOLD = 0.02    tiny but meaningful
  TSMOM_LOOKBACK = 42          (1 week)
  RSI_PERIOD = 2
  RSI_ENTRY = 10
  RSI_EXIT = 50

Run:
    uv run python scripts/backtest_regime_switch.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from crypto_algo.backtesting.simulator import run_backtest
from crypto_algo.data.loaders import load_parquet

INITIAL_EQUITY = 10_000.0
PERIODS_PER_YEAR = 6 * 365
COST_BPS = 5.0

# Frozen
AUTOCORR_WINDOW = 500
AUTOCORR_THRESHOLD = 0.02
TSMOM_LOOKBACK = 42
RSI_PERIOD = 2
RSI_ENTRY = 10.0
RSI_EXIT = 50.0

SYMBOLS = ["BTCUSDT", "ETHUSDT"]

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


def rsi_wilder(close: pd.Series, period: int = 2) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100.0 - 100.0 / (1.0 + rs)


def rolling_autocorr_lag1(returns: pd.Series, window: int) -> pd.Series:
    """Causal rolling lag-1 autocorrelation via rolling cov/var formula."""
    x = returns
    x_lag = returns.shift(1)
    mean = x.rolling(window, min_periods=window).mean()
    cov = ((x - mean) * (x_lag - mean)).rolling(window, min_periods=window).mean()
    var = x.rolling(window, min_periods=window).var()
    return (cov / var.replace(0, np.nan)).rename("autocorr")


def regime_series(returns: pd.Series) -> pd.Series:
    """Return regime label per bar: 'trend', 'range', or 'unclear'."""
    ac = rolling_autocorr_lag1(returns, AUTOCORR_WINDOW)
    regime = pd.Series("unclear", index=returns.index, dtype=object)
    regime[ac > AUTOCORR_THRESHOLD] = "trend"
    regime[ac < -AUTOCORR_THRESHOLD] = "range"
    return regime


def tsmom_signal(close: pd.Series, lookback: int = TSMOM_LOOKBACK) -> pd.Series:
    trail = np.log(close / close.shift(lookback))
    sig = (trail > 0).astype(float).fillna(0.0)
    return sig.rename("tsmom")


def reversion_signal(close: pd.Series) -> pd.Series:
    """RSI(2) state machine: enter long when < 10, exit when > 50."""
    rsi = rsi_wilder(close, RSI_PERIOD)
    n = len(close)
    state = np.zeros(n, dtype=float)
    in_pos = False
    for i in range(n):
        r = rsi.iloc[i]
        if pd.isna(r):
            state[i] = 1.0 if in_pos else 0.0
            continue
        if not in_pos and r < RSI_ENTRY:
            in_pos = True
        elif in_pos and r > RSI_EXIT:
            in_pos = False
        state[i] = 1.0 if in_pos else 0.0
    return pd.Series(state, index=close.index, name="rev")


def combined_signal(close: pd.Series, regime: pd.Series) -> pd.Series:
    """Combine: trend regime -> TSMOM, range regime -> reversion, unclear -> flat."""
    tsmom = tsmom_signal(close)
    rev = reversion_signal(close)

    combined = pd.Series(0.0, index=close.index)
    combined[regime == "trend"] = tsmom[regime == "trend"]
    combined[regime == "range"] = rev[regime == "range"]

    # Shift for next-bar execution.
    return combined.shift(1).fillna(0.0).rename("combined")


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


def _fmt(v: float, pct: bool = False) -> str:
    if v != v:
        return "n/a"
    if pct:
        return f"{v:+.2%}"
    return f"{v:+.3f}" if abs(v) < 100 else f"{v:+,.2f}"


def run_symbol(symbol: str, start: str, end: str, label: str) -> dict | None:
    path = Path(f"data/raw/binance/futures/{symbol}/4h/{start}_{end}.parquet")
    if not path.exists():
        print(f"  {symbol}: no data at {path}")
        return None

    df = load_parquet(path).reset_index(drop=True)
    close = df["close"]
    rets = np.log(close).diff()

    # Regime (causal, no shift yet)
    regime = regime_series(rets).reset_index(drop=True)

    # Signals
    tsmom = tsmom_signal(close).reset_index(drop=True)
    rev = reversion_signal(close).reset_index(drop=True)
    combined = combined_signal(close, regime).reset_index(drop=True)

    cost = _Cost(COST_BPS)

    # TSMOM baseline
    r_tsmom = run_backtest(df, tsmom, cost, INITIAL_EQUITY, PERIODS_PER_YEAR)
    s_tsmom = _stats(r_tsmom.equity)

    # Reversion baseline (always-on)
    r_rev = run_backtest(df, rev, cost, INITIAL_EQUITY, PERIODS_PER_YEAR)
    s_rev = _stats(r_rev.equity)

    # Regime switch
    r_sw = run_backtest(df, combined, cost, INITIAL_EQUITY, PERIODS_PER_YEAR)
    s_sw = _stats(r_sw.equity)

    # Regime occupancy
    n = len(regime)
    occ_trend = (regime == "trend").mean()
    occ_range = (regime == "range").mean()
    occ_unclear = (regime == "unclear").mean()

    # Regime switches: count transitions where label changed
    labels = regime.to_numpy()
    switches = int((labels[1:] != labels[:-1]).sum()) if n > 1 else 0

    # Per-regime performance of underlying signals
    # What did TSMOM do during trend regime bars? Reversion during range bars?
    daily_eq_tsmom = r_tsmom.equity.pct_change().fillna(0.0)
    daily_eq_rev = r_rev.equity.pct_change().fillna(0.0)

    trend_mask = (regime == "trend").to_numpy()
    range_mask = (regime == "range").to_numpy()

    tsmom_in_trend = daily_eq_tsmom[trend_mask]
    rev_in_range = daily_eq_rev[range_mask]

    def _regime_sharpe(r: pd.Series) -> float:
        if len(r) < 50:
            return float("nan")
        std = float(r.std())
        return float(r.mean() / std * np.sqrt(PERIODS_PER_YEAR)) if std > 0 else float("nan")

    return {
        "symbol": symbol,
        "label": label,
        "bars": n,
        "occ_trend": occ_trend,
        "occ_range": occ_range,
        "occ_unclear": occ_unclear,
        "switches": switches,
        "s_tsmom": s_tsmom,
        "s_rev": s_rev,
        "s_sw": s_sw,
        "tsmom_in_trend_sharpe": _regime_sharpe(tsmom_in_trend),
        "rev_in_range_sharpe": _regime_sharpe(rev_in_range),
    }


def main() -> None:
    for label, start, end in WINDOWS:
        print(f"\n{'=' * 88}")
        print(f"  Regime Switching — {label}")
        print(f"  Autocorr window = {AUTOCORR_WINDOW} bars | threshold = {AUTOCORR_THRESHOLD}")
        print(f"{'=' * 88}")

        for symbol in SYMBOLS:
            r = run_symbol(symbol, start, end, label)
            if r is None:
                continue

            print(f"\n  --- {symbol} ---")
            print(f"    Bars:            {r['bars']}")
            print(f"    Regime occupancy:")
            print(f"      trend:         {r['occ_trend']:.1%}")
            print(f"      range:         {r['occ_range']:.1%}")
            print(f"      unclear:       {r['occ_unclear']:.1%}")
            print(f"    Regime switches: {r['switches']} "
                  f"(~{r['switches'] / max(r['bars']/PERIODS_PER_YEAR, 1):.0f}/year)")

            print(f"\n    Strategy comparison:")
            print(f"      {'strategy':>12s}  {'total_ret':>10s}  {'sharpe':>8s}  {'max_dd':>9s}")
            for name, s in [
                ("TSMOM always", r["s_tsmom"]),
                ("Reversion always", r["s_rev"]),
                ("Regime switch", r["s_sw"]),
            ]:
                print(f"      {name:>12s}  "
                      f"{_fmt(s['total_return'], pct=True):>10s}  "
                      f"{_fmt(s['sharpe']):>8s}  "
                      f"{_fmt(s['max_dd'], pct=True):>9s}")

            print(f"\n    Per-regime diagnostic:")
            print(f"      TSMOM Sharpe during 'trend' regime:     {_fmt(r['tsmom_in_trend_sharpe'])}")
            print(f"      Reversion Sharpe during 'range' regime: {_fmt(r['rev_in_range_sharpe'])}")


if __name__ == "__main__":
    main()
    
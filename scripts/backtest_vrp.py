"""Deribit VRP (Volatility Risk Premium) backtest.

Hypothesis: implied vol (DVOL) systematically trades above realized vol.
Sell variance when the premium is wide, buy when inverted.

Simplified P&L model (variance swap proxy):
  At each day t, if signal fires:
    - K = DVOL(t) / 100 (decimal, annualized)
    - Over next 30 days, compute RV = realized vol from daily returns
    - Short P&L fraction = 0.2 × (K - RV)
    - Long P&L fraction = -0.2 × (K - RV)
  Position held exactly 30 days. New entries can overlap.

The 0.2 scaling factor converts vol-point difference to P&L, calibrated
to the literature (~1% per 30-day cycle at a 5-point premium). This is
a proxy for a vega-notional-matched variance swap.

Run:
    uv run python scripts/backtest_vrp.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

INITIAL_EQUITY = 10_000.0
HOLD_DAYS = 30
POSITION_FRACTION = 1.0    # fraction of equity per position
PAYOFF_SCALE = 0.2         # calibration constant (see docstring)

# Signal thresholds
SHORT_ENTRY_VRP = 0.03     # short vol if VRP > 3%
LONG_ENTRY_VRP = -0.03     # long vol if VRP < -3%

WINDOWS = [
    ("FULL 2024-2026",  "2023-12-29", "2026-09-23"),
    ("2025-2026 only",  "2025-01-01", "2026-09-23"),
]



def load_dvol(currency: str) -> pd.DataFrame:
    path = Path(f"data/raw/deribit/dvol/{currency}_2021-01-01_2026-09-23.parquet")
    if not path.exists():
        raise FileNotFoundError(f"DVOL file missing: {path}")
    df = pd.read_parquet(path)
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["date"] = pd.to_datetime(df["timestamp"]).dt.normalize()
    return df[["date", "close"]].rename(columns={"close": "dvol"}).dropna()


def load_spot(currency: str) -> pd.DataFrame:
    symbol = f"{currency}USDT"
    frames = []
    for suffix in ["2020-01-01_2025-01-01", "2025-01-01_2026-09-23"]:
        p = Path(f"data/raw/binance/futures/{symbol}/4h/{suffix}.parquet")
        if p.exists():
            frames.append(pd.read_parquet(p))
    if not frames:
        raise FileNotFoundError(f"No spot file for {symbol}")
    df = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["open_time"]).sort_values("open_time")
    df["date"] = pd.to_datetime(df["open_time"]).dt.normalize()
    daily = df.groupby("date")["close"].last().reset_index()
    return daily

def compute_realized_vol(spot: pd.DataFrame, window: int = 30, forward: bool = False) -> pd.Series:
    """Annualized realized vol over `window` days. If forward=True, uses future returns."""
    prices = spot.set_index("date")["close"]
    log_ret = np.log(prices).diff()
    if forward:
        # Realized vol over NEXT `window` days (from t+1 to t+window)
        # Reverse-shift to align
        fwd = log_ret.shift(-1).rolling(window).std().shift(-(window - 1))
        return fwd * np.sqrt(365)
    else:
        return log_ret.rolling(window).std() * np.sqrt(365)


def build_dataset(currency: str) -> pd.DataFrame:
    dvol = load_dvol(currency)
    spot = load_spot(currency)

    merged = pd.merge(dvol, spot, on="date", how="inner").sort_values("date").reset_index(drop=True)

    if len(merged) < 50:
        return merged

    # Compute RV directly on the merged frame — no reindex, no dtype issues.
    prices = merged.set_index("date")["close"]
    log_ret = np.log(prices).diff()

    # Trailing 30-day realized vol (annualized).
    rv_trailing = log_ret.rolling(30).std() * np.sqrt(365)

    # Forward 30-day realized vol: returns from t+1 to t+30.
    # Equivalent: shift returns back by 1, then rolling std.
    log_ret_fwd = log_ret.shift(-1)
    rv_forward = log_ret_fwd.rolling(30).std().shift(-29) * np.sqrt(365)

    merged["rv_trailing"] = rv_trailing.values
    merged["rv_forward"] = rv_forward.values
    merged["dvol_dec"] = merged["dvol"] / 100.0
    merged["vrp"] = merged["dvol_dec"] - merged["rv_trailing"]

    return merged.dropna(subset=["dvol_dec", "rv_trailing", "rv_forward"]).reset_index(drop=True)


def backtest(df: pd.DataFrame, label: str) -> dict:
    """Non-overlapping: open a position, hold 30 days, then wait for next signal.

    P&L is applied at exit, not daily. No compounding bug.
    """
    equity = INITIAL_EQUITY
    curve = []
    positions = []
    n_short = 0
    n_long = 0
    n_skipped = 0
    next_available = df["date"].iloc[0]  # earliest date we can open a new position

    for _, row in df.iterrows():
        equity_today = equity
        direction = 0

        if row["date"] >= next_available:
            vrp = row["vrp"]
            if vrp > SHORT_ENTRY_VRP:
                direction = 1
                n_short += 1
            elif vrp < LONG_ENTRY_VRP:
                direction = -1
                n_long += 1
            else:
                direction = 0
                n_skipped += 1

            if direction != 0:
                payoff_frac = direction * PAYOFF_SCALE * (row["dvol_dec"] - row["rv_forward"])
                pnl = equity * payoff_frac
                # Apply the full 30-day payoff at exit time.
                exit_date = row["date"] + pd.Timedelta(days=HOLD_DAYS)
                positions.append({
                    "entry": row["date"],
                    "exit": exit_date,
                    "direction": direction,
                    "payoff_frac": payoff_frac,
                    "pnl": pnl,
                })
                equity += pnl
                next_available = exit_date

        curve.append({"date": row["date"], "equity": equity})

    curve_df = pd.DataFrame(curve).set_index("date")

    rets = curve_df["equity"].pct_change().dropna()
    if len(rets) < 2:
        return {"label": label, "error": "insufficient data"}

    total = float(curve_df["equity"].iloc[-1] / INITIAL_EQUITY - 1.0)
    years = (curve_df.index[-1] - curve_df.index[0]).days / 365.25
    cagr = float((curve_df["equity"].iloc[-1] / INITIAL_EQUITY) ** (1 / years) - 1) if years > 0 else float("nan")
    std = float(rets.std())
    sharpe = float(rets.mean() / std * np.sqrt(365)) if std > 0 else float("nan")
    peak = curve_df["equity"].cummax()
    max_dd = float((curve_df["equity"] / peak - 1.0).min())

    wins = sum(1 for p in positions if p["pnl"] > 0)
    n_pos = len(positions)

    return {
        "label": label,
        "start": curve_df.index[0].date(),
        "end": curve_df.index[-1].date(),
        "years": years,
        "final_equity": float(curve_df["equity"].iloc[-1]),
        "total_return": total,
        "cagr": cagr,
        "sharpe": sharpe,
        "max_dd": max_dd,
        "n_short": n_short,
        "n_long": n_long,
        "n_skipped": n_skipped,
        "n_positions": n_pos,
        "win_rate": wins / n_pos if n_pos > 0 else float("nan"),
    }



def _fmt(v: float, pct: bool = False) -> str:
    if v != v:
        return "n/a"
    if pct:
        return f"{v:+.2%}"
    return f"{v:+.3f}" if abs(v) < 100 else f"{v:+,.2f}"


def main() -> None:
    for currency in ["BTC", "ETH"]:
        print(f"\n{'=' * 78}")
        print(f"  VRP Backtest — {currency}  (Deribit DVOL vs realized vol)")
        print(f"{'=' * 78}")

        df = build_dataset(currency)
        if len(df) < 100:
            print(f"  Insufficient data: {len(df)} rows")
            continue

        print(f"  Data range: {df['date'].iloc[0].date()} to {df['date'].iloc[-1].date()}  ({len(df)} days)")
        print(f"  DVOL mean: {df['dvol'].mean():.1f}  |  RV mean: {(df['rv_trailing'] * 100).mean():.1f}")

        for label, start, end in WINDOWS:
            start_ts = pd.Timestamp(start, tz="UTC")
            end_ts = pd.Timestamp(end, tz="UTC")
            sub = df[(df["date"] >= start_ts) & (df["date"] <= end_ts)].reset_index(drop=True)
            if len(sub) < 60:
                print(f"\n  [{label}] insufficient data ({len(sub)} rows)")
                continue
            result = backtest(sub, label)

            print(f"\n  [{label}]  {result['start']} to {result['end']}  ({result['years']:.2f} years)")
            print(f"    Final equity:      ${result['final_equity']:>10,.2f}")
            print(f"    Total return:      {_fmt(result['total_return'], pct=True)}")
            print(f"    CAGR:              {_fmt(result['cagr'], pct=True)}")
            print(f"    Sharpe:            {_fmt(result['sharpe'])}")
            print(f"    Max drawdown:      {_fmt(result['max_dd'], pct=True)}")
            print(f"    Positions:         {result['n_positions']}  "
                  f"(short={result['n_short']}, long={result['n_long']}, "
                  f"flat={result['n_skipped']})")
            print(f"    Win rate:          {_fmt(result['win_rate'], pct=True)}")



    # Also print the raw premium statistics.
    for currency in ["BTC", "ETH"]:
        print(f"\n{'=' * 78}")
        print(f"  RAW PREMIUM ANALYSIS — {currency}")
        print(f"{'=' * 78}")
        try:
            df = build_dataset(currency)
        except FileNotFoundError as e:
            print(f"  skipped: {e}")
            continue
        if len(df) < 60:
            print(f"  insufficient data: {len(df)}")
            continue
        prem = df["dvol_dec"] - df["rv_forward"]
        n = len(prem)
        mean = float(prem.mean())
        std = float(prem.std())
        se = std / np.sqrt(n)
        t = mean / se if se > 0 else float("nan")
        print(f"  n days:              {n}")
        print(f"  Mean premium:        {mean * 100:+.2f}% annualized")
        print(f"  Median:              {prem.median() * 100:+.2f}%")
        print(f"  Std:                 {std * 100:.2f}%")
        print(f"  % positive:          {(prem > 0).mean():.1%}")
        print(f"  t-stat vs 0:         {t:+.2f}")
        if abs(t) > 2.0 and mean > 0:
            print(f"  VERDICT: positive premium with statistical significance")
        elif mean > 0:
            print(f"  VERDICT: positive premium, not statistically significant")
        else:
            print(f"  VERDICT: no positive premium")
            
if __name__ == "__main__":
    main()

"""Signal scan — systematic test of input predictive power.

Tests three candidate inputs against forward returns at 3 horizons:
  1. taker_flow  = taker_buy_base_volume / volume  (aggressive buy ratio)
  2. funding_z   = rolling 30-event z-score of funding rate
  3. vol_ratio   = current vol / trailing median vol  (baseline)

For each (input, horizon, symbol): compute Spearman correlation.
Aggregate across symbols with a one-sample t-test on the per-symbol
correlations.

Bonferroni: 3 inputs × 3 horizons × 2 windows = 18 tests.
Threshold α = 0.05/18 ≈ 0.0028. t_crit (df=18) ≈ 3.6.

Run:
    uv run python scripts/signal_scan.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from crypto_algo.data.loaders import load_parquet

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

HORIZONS = [1, 4, 24]   # 4H, 16H, 4 days
VOL_FAST = 30
VOL_SLOW = 200
FUNDING_WINDOW = 30

T_CRIT = 3.6   # Bonferroni-corrected t-threshold


def spearman(x: pd.Series, y: pd.Series) -> tuple[float, int]:
    """Spearman correlation via Pearson on ranks (no scipy dependency)."""
    df = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(df) < 100:
        return float("nan"), 0
    xr = df["x"].rank()
    yr = df["y"].rank()
    # Pearson correlation of ranks == Spearman correlation
    corr = float(xr.corr(yr))
    return corr, len(df)


def load_funding(symbol: str, start: str, end: str) -> pd.DataFrame | None:
    path = Path(f"data/raw/binance/funding/{symbol}/{start}_{end}.parquet")
    if not path.exists():
        return None
    return pd.read_parquet(path)


def compute_inputs(df: pd.DataFrame, funding_df: pd.DataFrame | None) -> dict[str, pd.Series]:
    inputs: dict[str, pd.Series] = {}

    # 1. Taker flow imbalance
    vol = df["volume"].replace(0, np.nan)
    inputs["taker_flow"] = (df["taker_buy_base_volume"] / vol).fillna(0.5)

    # 2. Vol ratio (current vol / trailing median vol)
    log_ret = np.log(df["close"]).diff()
    rv = log_ret.rolling(VOL_FAST, min_periods=VOL_FAST).std()
    rv_median = rv.rolling(VOL_SLOW, min_periods=VOL_FAST).median()
    inputs["vol_ratio"] = rv / rv_median.replace(0, np.nan)

    # 3. Funding z-score (shifted by 1 event to avoid contemporaneous leakage)
    if funding_df is not None and len(funding_df) > FUNDING_WINDOW + 5:
        f = funding_df[["funding_time", "funding_rate"]].copy().sort_values("funding_time")
        left = df[["open_time"]].reset_index(drop=True)
        merged = pd.merge_asof(
            left, f,
            left_on="open_time", right_on="funding_time",
            direction="backward",
        )
        fr = merged["funding_rate"].reset_index(drop=True).shift(1)
        mean = fr.rolling(FUNDING_WINDOW, min_periods=FUNDING_WINDOW).mean()
        std = fr.rolling(FUNDING_WINDOW, min_periods=FUNDING_WINDOW).std()
        inputs["funding_z"] = (fr - mean) / std.replace(0, np.nan)
    else:
        inputs["funding_z"] = pd.Series([np.nan] * len(df))

    return inputs


def forward_log_return(close: pd.Series, h: int) -> pd.Series:
    return np.log(close.shift(-h) / close)


def scan_one(symbol: str, start: str, end: str) -> dict:
    path = Path(f"data/raw/binance/futures/{symbol}/4h/{start}_{end}.parquet")
    if not path.exists():
        return {}
    df = load_parquet(path).reset_index(drop=True)
    funding_df = load_funding(symbol, start, end)
    inputs = compute_inputs(df, funding_df)
    close = df["close"]

    results = {}
    for name, series in inputs.items():
        series = series.reset_index(drop=True)
        for h in HORIZONS:
            fwd = forward_log_return(close, h)
            corr, n = spearman(series, fwd)
            results[(name, h)] = (corr, n)
    return results


def aggregate(corrs: list[float]) -> tuple[float, float, int]:
    vals = [c for c in corrs if c == c]
    n = len(vals)
    if n < 5:
        return float("nan"), float("nan"), n
    m = float(np.mean(vals))
    s = float(np.std(vals, ddof=1))
    t = m / (s / np.sqrt(n)) if s > 0 else float("nan")
    return m, t, n


def main() -> None:
    for window_label, start, end in WINDOWS:
        print(f"\n{'=' * 78}")
        print(f"  SIGNAL SCAN — {window_label}")
        print(f"{'=' * 78}")

        all_results: dict = {}
        for symbol in SYMBOLS:
            all_results[symbol] = scan_one(symbol, start, end)

        print(f"\n  {'input':>12s} {'h':>4s}  "
              f"{'mean_corr':>11s} {'t_stat':>8s} {'n_sym':>6s}  "
              f"{'result':>10s}")
        print(f"  {'-' * 62}")

        for input_name in ["taker_flow", "funding_z", "vol_ratio"]:
            for h in HORIZONS:
                corrs = []
                for symbol in SYMBOLS:
                    r = all_results.get(symbol, {})
                    if (input_name, h) in r:
                        corr, _ = r[(input_name, h)]
                        if corr == corr:
                            corrs.append(corr)
                m, t, n = aggregate(corrs)
                if t != t:
                    result = "--"
                elif abs(t) >= T_CRIT:
                    result = "PASS"
                else:
                    result = "no"
                print(f"  {input_name:>12s} {h:>4d}  "
                      f"{m:>+11.4f} {t:>+8.2f} {n:>6d}  {result:>10s}")

        # Per-symbol detail for h=1
        print(f"\n  Per-symbol correlations (h=1):")
        print(f"  {'symbol':>10s}  {'taker_flow':>10s}  {'funding_z':>10s}  {'vol_ratio':>10s}")
        for symbol in SYMBOLS:
            r = all_results.get(symbol, {})
            def g(k):
                v = r.get(k, (float("nan"), 0))[0]
                return f"{v:>+10.4f}" if v == v else "       n/a"
            print(f"  {symbol:>10s}  {g(('taker_flow', 1))}  "
                  f"{g(('funding_z', 1))}  {g(('vol_ratio', 1))}")


if __name__ == "__main__":
    main()

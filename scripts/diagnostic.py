"""Market diagnostic study.

Statistical tests on existing 4H data to determine what kind of market
we're in, comparing historical (2020-2025) to recent (2025-2026).

Tests:
  1. Return autocorrelation — trending vs mean-reverting
  2. Variance ratio — random walk vs trending vs reverting
  3. Volatility autocorrelation — is vol predictable?
  4. Cross-sectional dispersion — do alts diverge?
  5. BTC-ETH rolling correlation — is the pair stable?

Run:
    uv run python scripts/diagnostic.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

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

LAGS = [1, 4, 12, 24, 100]
VR_QS = [2, 4, 8, 24, 96]


def load_prices(symbol: str, start: str, end: str) -> pd.Series | None:
    path = Path(f"data/raw/binance/futures/{symbol}/4h/{start}_{end}.parquet")
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    df = df.sort_values("open_time").set_index("open_time")
    s = df["close"].rename(symbol)
    if len(s) < 200:
        return None
    return s


def load_all(start: str, end: str) -> pd.DataFrame:
    series = {}
    for s in SYMBOLS:
        col = load_prices(s, start, end)
        if col is not None:
            series[s] = col
    wide = pd.DataFrame(series).sort_index()
    return wide


def print_header(title: str) -> None:
    print(f"\n{'=' * 76}")
    print(f"  {title}")
    print(f"{'=' * 76}")


def return_autocorrelation(returns: pd.Series, lags: list[int]) -> dict:
    out = {}
    for lag in lags:
        try:
            out[lag] = returns.autocorr(lag=lag)
        except Exception:
            out[lag] = float("nan")
    return out


def variance_ratio(returns: pd.Series, q: int) -> float:
    """Lo-MacKinlay variance ratio. VR > 1 = trending, < 1 = mean-reverting."""
    n = len(returns)
    if n < q * 10:
        return float("nan")
    var_1 = returns.var()
    if var_1 <= 0:
        return float("nan")
    # q-period overlapping returns
    ret_q = returns.rolling(q).sum().dropna()
    var_q = ret_q.var()
    return float(var_q / (q * var_1))


def test_returns(prices: pd.DataFrame, label: str) -> None:
    # Portfolio-level: use BTC and ETH log returns as reference
    print_header(f"{label} — return autocorrelation (BTC, ETH, SOL)")
    print(f"  {'symbol':>10s}  " + "  ".join(f"lag={l:>3d}" for l in LAGS))
    for sym in ["BTCUSDT", "ETHUSDT", "SOLUSDT"]:
        if sym not in prices.columns:
            continue
        close = prices[sym].dropna()
        rets = np.log(close).diff().dropna()
        ac = return_autocorrelation(rets, LAGS)
        vals = "  ".join(f"{ac[l]:>+7.4f}" if ac[l] == ac[l] else "    n/a " for l in LAGS)
        print(f"  {sym:>10s}  {vals}")

    print_header(f"{label} — variance ratio")
    print(f"  {'symbol':>10s}  " + "  ".join(f"q={q:>3d}" for q in VR_QS))
    for sym in ["BTCUSDT", "ETHUSDT", "SOLUSDT"]:
        if sym not in prices.columns:
            continue
        close = prices[sym].dropna()
        rets = np.log(close).diff().dropna()
        vals = []
        for q in VR_QS:
            vr = variance_ratio(rets, q)
            vals.append(f"{vr:>6.3f}" if vr == vr else "   n/a")
        print(f"  {sym:>10s}  " + "  ".join(vals))


def test_vol_autocorrelation(prices: pd.DataFrame, label: str) -> None:
    print_header(f"{label} — volatility autocorrelation (|log return|)")
    print(f"  {'symbol':>10s}  " + "  ".join(f"lag={l:>3d}" for l in LAGS))
    for sym in ["BTCUSDT", "ETHUSDT", "SOLUSDT"]:
        if sym not in prices.columns:
            continue
        close = prices[sym].dropna()
        rets = np.log(close).diff().dropna()
        abs_rets = rets.abs()
        ac = return_autocorrelation(abs_rets, LAGS)
        vals = "  ".join(f"{ac[l]:>+7.4f}" if ac[l] == ac[l] else "    n/a " for l in LAGS)
        print(f"  {sym:>10s}  {vals}")


def test_cross_sectional_dispersion(prices: pd.DataFrame, label: str) -> None:
    print_header(f"{label} — cross-sectional dispersion")
    rets = np.log(prices).diff().dropna(how="all")
    # At each bar, std across symbols
    disp = rets.std(axis=1).dropna()
    print(f"  Mean 4H cross-sectional return std:  {disp.mean() * 1e4:.2f} bps")
    print(f"  Median:                              {disp.median() * 1e4:.2f} bps")
    print(f"  10th percentile:                     {disp.quantile(0.10) * 1e4:.2f} bps")
    print(f"  90th percentile:                     {disp.quantile(0.90) * 1e4:.2f} bps")

    # Rolling 30-day average dispersion
    disp_daily = disp.resample("1D").mean().dropna()
    if len(disp_daily) > 90:
        first = disp_daily.iloc[:90].mean() * 1e4
        last = disp_daily.iloc[-90:].mean() * 1e4
        print(f"  First 90 days mean:                  {first:.2f} bps")
        print(f"  Last 90 days mean:                   {last:.2f} bps")
        print(f"  Change:                              {(last/first - 1) * 100:+.1f}%")


def test_btc_eth_correlation(prices: pd.DataFrame, label: str) -> None:
    print_header(f"{label} — BTC-ETH correlation")
    if "BTCUSDT" not in prices.columns or "ETHUSDT" not in prices.columns:
        print("  Missing BTC or ETH.")
        return
    btc = np.log(prices["BTCUSDT"]).diff()
    eth = np.log(prices["ETHUSDT"]).diff()
    df = pd.DataFrame({"btc": btc, "eth": eth}).dropna()
    corr = df["btc"].corr(df["eth"])
    print(f"  Full-period correlation:       {corr:+.4f}")
    # Rolling 500-bar correlation
    rolling = df["btc"].rolling(500).corr(df["eth"]).dropna()
    if len(rolling) > 0:
        print(f"  Rolling 500-bar min/mean/max:  {rolling.min():+.3f} / {rolling.mean():+.3f} / {rolling.max():+.3f}")
        print(f"  Rolling last 500-bar value:    {rolling.iloc[-1]:+.3f}")


def main() -> None:
    for label, start, end in WINDOWS:
        print(f"\n\n{'#' * 76}")
        print(f"#  {label}   ({start} to {end})")
        print(f"{'#' * 76}")

        prices = load_all(start, end)
        if prices.empty:
            print(f"  No data for {start} to {end}")
            continue

        print(f"  Symbols loaded: {prices.shape[1]}")
        print(f"  Bars:           {prices.shape[0]}")
        print(f"  Date range:     {prices.index[0]} to {prices.index[-1]}")

        test_returns(prices, label)
        test_vol_autocorrelation(prices, label)
        test_cross_sectional_dispersion(prices, label)
        test_btc_eth_correlation(prices, label)


if __name__ == "__main__":
    main()
    
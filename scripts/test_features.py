"""Prove features are causal: altering future rows does not change past features."""

from __future__ import annotations

import numpy as np
import pandas as pd

from crypto_algo.features.engine import build_features


def main() -> None:
    n = 500
    rng = np.random.default_rng(42)
    prices = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    times = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")

    df = pd.DataFrame({
        "open_time": times,
        "close_time": times + pd.Timedelta(minutes=59, seconds=59, milliseconds=999),
        "open": prices,
        "high": prices * 1.001,
        "low": prices * 0.999,
        "close": prices,
        "volume": rng.uniform(100, 1000, n),
        "quote_volume": rng.uniform(10_000, 100_000, n),
        "num_trades": rng.integers(100, 1000, n),
        "taker_buy_base_volume": rng.uniform(50, 500, n),
        "taker_buy_quote_volume": rng.uniform(5_000, 50_000, n),
        "symbol": "TESTUSDT",
        "interval": "1h",
        "source": "test",
    })

    cfg = {"ma_fast": 20, "ma_slow": 100}
    features_orig = build_features(df, cfg)

    # Modify ONLY future rows (row 300 onward).
    df_mod = df.copy()
    df_mod.loc[300:, "close"] = df_mod.loc[300:, "close"] * 10.0
    df_mod.loc[300:, "open"] = df_mod.loc[300:, "open"] * 10.0
    df_mod.loc[300:, "high"] = df_mod.loc[300:, "high"] * 10.0
    df_mod.loc[300:, "low"] = df_mod.loc[300:, "low"] * 10.0

    features_mod = build_features(df_mod, cfg)

    cols = ["ret_1", "ret_24", "vol_24", "vol_168", "ma_fast", "ma_slow", "zscore_24"]
    for col in cols:
        a = features_orig[col].iloc[:300]
        b = features_mod[col].iloc[:300]
        pd.testing.assert_series_equal(a, b, check_names=False)

    print("Feature causality verified: modifying future data does not change past features.")


if __name__ == "__main__":
    main()

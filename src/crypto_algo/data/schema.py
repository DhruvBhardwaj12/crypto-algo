"""Canonical OHLCV schema enforced across the entire system.

Every DataFrame that enters the strategy pipeline must pass check_schema().
This keeps downstream code simple: features, strategies, and backtests
can assume the schema is stable and typed correctly.
"""

from __future__ import annotations

import pandas as pd

REQUIRED_COLUMNS: list[str] = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
    "num_trades",
    "taker_buy_base_volume",
    "taker_buy_quote_volume",
    "symbol",
    "interval",
    "source",
]

NUMERIC_COLUMNS: list[str] = [
    "open",
    "high",
    "low",
    "close",
    "volume",
    "quote_volume",
    "taker_buy_base_volume",
    "taker_buy_quote_volume",
]

TIMESTAMP_COLUMNS: list[str] = ["open_time", "close_time"]


def check_schema(df: pd.DataFrame) -> None:
    """Raise ValueError/TypeError if df does not match the canonical schema.

    We fail loudly here. Silent acceptance of malformed data is the
    single biggest source of bogus backtest results.
    """
    if df.empty:
        raise ValueError("DataFrame is empty.")

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    for col in TIMESTAMP_COLUMNS:
        if not pd.api.types.is_datetime64_any_dtype(df[col]):
            raise TypeError(f"Column '{col}' must be datetime, got {df[col].dtype}")
        if df[col].dt.tz is None:
            raise ValueError(
                f"Column '{col}' must be timezone-aware (UTC). Naive timestamps "
                "are a common source of silent backtest errors."
            )

    for col in NUMERIC_COLUMNS:
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise TypeError(f"Column '{col}' must be numeric, got {df[col].dtype}")

    if not pd.api.types.is_integer_dtype(df["num_trades"]):
        raise TypeError(
            f"Column 'num_trades' must be integer, got {df['num_trades'].dtype}"
        )
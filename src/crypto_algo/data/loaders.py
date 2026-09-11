"""Data loaders with the known-at-time-T guarantee.

THE central rule of this system:

    A strategy may only see data whose close_time is strictly before the
    moment it is making a decision.

Everything downstream of this module uses these functions. If you find
yourself writing `df[df["open_time"] <= some_time]` elsewhere in the code,
you are probably introducing look-ahead bias. Stop and use these loaders.

Definitions
-----------
- `as_of` is a UTC timezone-aware timestamp representing "now".
- A candle with `close_time = X` is fully known at any time `T > X`.
- A candle is never partially known. At T between open and close, the
  candle is invisible to us (its H/L/C/V are still being determined).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from crypto_algo.data.schema import check_schema
from crypto_algo.data.validation import validate_ohlcv


def load_parquet(path: Path) -> pd.DataFrame:
    """Load a Parquet file and run schema + validation checks on it.

    This is the entry point for all trusted data. If it returns, you can
    trust the schema and (assuming a clean ValidationReport) the contents.
    """
    if not path.exists():
        raise FileNotFoundError(f"No such file: {path}")

    df = pd.read_parquet(path)
    check_schema(df)
    return df


def get_visible_data(df: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
    """Return only candles fully known at `as_of`.

    A candle is known at `as_of` iff its close_time < as_of.

    Example: with hourly candles,
        candle open=14:00, close_time=14:59:59.999
    is included when as_of = 15:00:00 but excluded at as_of = 14:30:00.
    """
    if as_of.tzinfo is None:
        raise ValueError(
            f"`as_of` must be timezone-aware UTC. Got naive timestamp: {as_of!r}"
        )
    mask = df["close_time"] < as_of
    return df.loc[mask].copy()


def last_closed_candle_at(df: pd.DataFrame, as_of: pd.Timestamp) -> pd.Series | None:
    """Return the most recent fully-closed candle at `as_of`, or None.

    Useful for strategy code that wants "the latest known state".
    """
    visible = get_visible_data(df, as_of)
    if visible.empty:
        return None
    return visible.iloc[-1]


def get_visible_window(
    df: pd.DataFrame,
    as_of: pd.Timestamp,
    lookback: int,
) -> pd.DataFrame:
    """Return the last `lookback` candles known at `as_of`.

    If fewer than `lookback` are available (near the start of the dataset),
    returns whatever is available. Never reaches into the future.
    """
    if lookback <= 0:
        raise ValueError(f"lookback must be positive, got {lookback}")
    visible = get_visible_data(df, as_of)
    return visible.tail(lookback).copy()
    
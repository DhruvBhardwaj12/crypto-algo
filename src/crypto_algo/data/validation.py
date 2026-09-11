"""Structural validation and gap detection for OHLCV data.

Validation is deliberately read-only: it inspects data and returns a report.
It never fixes, fills, or mutates. Deciding what to do about a problem is a
separate concern handled downstream (in the feature engine or the strategy).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from crypto_algo.data.schema import check_schema


@dataclass
class ValidationReport:
    """Summary of structural checks on one symbol's dataset."""

    symbol: str
    interval: str
    rows: int
    start: pd.Timestamp
    end: pd.Timestamp
    n_duplicates: int
    n_nan: int
    n_gaps: int
    missing_intervals: list[pd.Timestamp] = field(default_factory=list)
    ohlc_violations: int = 0
    negative_price_or_volume: int = 0

    @property
    def is_clean(self) -> bool:
        return (
            self.n_duplicates == 0
            and self.n_nan == 0
            and self.n_gaps == 0
            and self.ohlc_violations == 0
            and self.negative_price_or_volume == 0
        )

    def summary(self) -> str:
        lines = [
            f"ValidationReport({self.symbol} {self.interval})",
            f"  rows:               {self.rows}",
            f"  range:              {self.start} -> {self.end}",
            f"  duplicates:         {self.n_duplicates}",
            f"  NaN cells:          {self.n_nan}",
            f"  gaps:               {self.n_gaps}",
            f"  OHLC violations:    {self.ohlc_violations}",
            f"  negative p/v:       {self.negative_price_or_volume}",
            f"  clean:              {self.is_clean}",
        ]
        return "\n".join(lines)


def _expected_grid(start: pd.Timestamp, end: pd.Timestamp, interval: str) -> pd.DatetimeIndex:
    """Build the exact index of timestamps a complete series would have."""
    return pd.date_range(start=start, end=end, freq=interval, tz="UTC")


def validate_ohlcv(df: pd.DataFrame, interval: str = "1h") -> ValidationReport:
    """Run all structural checks. Does not modify the DataFrame."""
    check_schema(df)

    if len(df) == 0:
        raise ValueError("Cannot validate empty DataFrame.")

    # Duplicates
    n_duplicates = int(df["open_time"].duplicated().sum())

    # NaN
    n_nan = int(df.isna().sum().sum())

    # Monotonicity (warning-only check; we don't raise here)
    if not df["open_time"].is_monotonic_increasing:
        # If unsorted, sort a copy just for gap analysis. We do NOT mutate input.
        df = df.sort_values("open_time")

    # Gap analysis
    start = df["open_time"].min()
    end = df["open_time"].max()
    expected = _expected_grid(start, end, interval)
    actual = pd.DatetimeIndex(df["open_time"])
    missing = expected.difference(actual)

    # OHLC violations: high must be >= all of open/close/low; low <= all.
    high_ok = df["high"] >= df[["open", "close", "low"]].max(axis=1)
    low_ok = df["low"] <= df[["open", "close", "high"]].min(axis=1)
    ohlc_violations = int((~high_ok).sum() + (~low_ok).sum())

    # Negative prices/volumes
    neg = (
        (df[["open", "high", "low", "close"]] <= 0).sum().sum()
        + (df[["volume", "quote_volume"]] < 0).sum().sum()
    )
    negative = int(neg)

    return ValidationReport(
        symbol=str(df["symbol"].iloc[0]),
        interval=str(df["interval"].iloc[0]),
        rows=len(df),
        start=start,
        end=end,
        n_duplicates=n_duplicates,
        n_nan=n_nan,
        n_gaps=len(missing),
        missing_intervals=list(missing),
        ohlc_violations=ohlc_violations,
        negative_price_or_volume=negative,
    )
    
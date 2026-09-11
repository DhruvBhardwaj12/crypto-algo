"""Prove the known-at-time-T guarantee behaves as expected."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from crypto_algo.data.loaders import get_visible_data, last_closed_candle_at, load_parquet


def main() -> None:
    path = Path("data/raw/binance/BTCUSDT/1h/2023-01-01_2024-01-01.parquet")
    df = load_parquet(path)

    as_of = pd.Timestamp("2023-06-15 00:00:00", tz="UTC")

    visible = get_visible_data(df, as_of)
    last = last_closed_candle_at(df, as_of)

    print(f"as_of:            {as_of}")
    print(f"rows total:       {len(df)}")
    print(f"rows visible:     {len(visible)}")
    print(f"first visible:    {visible['open_time'].iloc[0]}")
    print(f"last visible:     {visible['open_time'].iloc[-1]}")
    print(f"last close_time:  {visible['close_time'].iloc[-1]}")
    print()
    assert last is not None
    print(f"last closed candle open_time:  {last['open_time']}")
    print(f"last closed candle close_time: {last['close_time']}")
    print(f"last closed candle close:      {last['close']}")

    assert (visible["close_time"] < as_of).all(), (
        "Look-ahead leak: some visible row closes at or after as_of"
    )
    assert visible["open_time"].iloc[-1] == pd.Timestamp("2023-06-14 23:00", tz="UTC"), (
        f"Expected last visible candle open_time = 2023-06-14 23:00, "
        f"got {visible['open_time'].iloc[-1]}"
    )
    next_row = df[df["open_time"] == pd.Timestamp("2023-06-15 00:00", tz="UTC")]
    assert not next_row.empty, "Test setup error: next candle not present in raw data"
    assert next_row["open_time"].iloc[0] not in visible["open_time"].values, (
        "Look-ahead leak: candle that should be invisible is present in visible data"
    )

    print()
    print("Visibility guarantee verified.")


if __name__ == "__main__":
    main()
    
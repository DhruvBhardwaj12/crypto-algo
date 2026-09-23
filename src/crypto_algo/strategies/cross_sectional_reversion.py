"""Cross-sectional mean reversion.

Opposite of LAB-007 (cross-sectional momentum). Rank the universe by
trailing return. SHORT the top performers, LONG the worst performers.
Bet that recent dispersion reverts.

At each rebalance:
  - Compute trailing return over `lookback` bars for every available symbol
  - Rank
  - Short the top n_short, long the bottom n_long
  - Weights: -0.5/n_short for shorts, +0.5/n_long for longs (gross = 1.0)
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def cross_sectional_reversion_weights(
    prices: pd.DataFrame,
    lookback: int = 42,
    n_long: int = 3,
    n_short: int = 3,
    rebalance_every: int = 6,
) -> pd.DataFrame:
    if lookback < 2:
        raise ValueError("lookback must be >= 2")
    if n_long < 1 or n_short < 1:
        raise ValueError("n_long and n_short must be >= 1")

    trailing = np.log(prices / prices.shift(lookback))
    weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

    long_weight = 0.5 / n_long
    short_weight = -0.5 / n_short

    n = len(prices)
    for i in range(lookback, n, rebalance_every):
        row = trailing.iloc[i]
        available = row.dropna()
        if len(available) < n_long + n_short:
            continue

        ranked = available.sort_values(ascending=False)  # best first
        winners = ranked.head(n_short).index.tolist()   # short these
        losers = ranked.tail(n_long).index.tolist()      # long these

        end = min(i + rebalance_every, n)
        for s in winners:
            col = weights.columns.get_loc(s)
            weights.iloc[i:end, col] = short_weight
        for s in losers:
            col = weights.columns.get_loc(s)
            weights.iloc[i:end, col] = long_weight

    weights = weights.where(prices.notna(), 0.0)
    return weights

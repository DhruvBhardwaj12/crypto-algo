"""Cross-sectional momentum: rank the universe, long winners, short losers.

At each rebalance, compute trailing return over `lookback` bars for every
available symbol. Rank. Long the top `n_long`, short the bottom `n_short`.
Weights are equal, with gross exposure = 1.0.

Weights persist between rebalances. Weights are zero for symbols with no
price data at the current bar (dynamic universe).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def cross_sectional_momentum_weights(
    prices: pd.DataFrame,
    lookback: int = 42,
    n_long: int = 3,
    n_short: int = 3,
    rebalance_every: int = 6,
) -> pd.DataFrame:
    """Return a weight matrix (same shape as prices) with values in [-1, 1].

    Parameters
    ----------
    prices : wide close prices, index=time, columns=symbols
    lookback : trailing return window in bars
    n_long : number of top-ranked symbols to long
    n_short : number of bottom-ranked symbols to short
    rebalance_every : number of bars between rebalances
    """
    if lookback < 2:
        raise ValueError("lookback must be >= 2")
    if n_long < 1 or n_short < 1:
        raise ValueError("n_long and n_short must be >= 1")

    trailing = np.log(prices / prices.shift(lookback))

    weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

    # Per-side weight. Longs sum to +0.5, shorts to -0.5. Gross = 1.0.
    per_side_weight = 0.5 / max(n_long, n_short)

    n = len(prices)
    for i in range(lookback, n, rebalance_every):
        row = trailing.iloc[i]
        available = row.dropna()
        if len(available) < n_long + n_short:
            continue

        ranked = available.sort_values(ascending=False)
        longs = ranked.head(n_long).index.tolist()
        shorts = ranked.tail(n_short).index.tolist()

        # Fill forward until the next rebalance.
        end = min(i + rebalance_every, n)
        for s in longs:
            col = weights.columns.get_loc(s)
            weights.iloc[i:end, col] = per_side_weight
        for s in shorts:
            col = weights.columns.get_loc(s)
            weights.iloc[i:end, col] = -per_side_weight

    # Zero out weights where price is missing.
    weights = weights.where(prices.notna(), 0.0)

    return weights

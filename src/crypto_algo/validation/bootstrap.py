"""Bootstrap confidence intervals and significance tests.

Given a daily return series, we estimate the sampling distribution of the
Sharpe ratio via block bootstrap. If the 5th percentile is below zero, we
cannot distinguish the observed Sharpe from luck.

We also run a binomial sign test on the trade P&L: under a null of no edge,
the win rate should be 50%.
"""

from __future__ import annotations

from math import comb

import numpy as np
import pandas as pd


def block_bootstrap_sharpe(
    returns: pd.Series,
    block_size: int = 30,
    n_resamples: int = 2000,
    periods_per_year: int = 365,
    seed: int = 42,
) -> dict:
    """Moving-block bootstrap of the Sharpe ratio.

    Standard IID bootstrap destroys autocorrelation (which returns have).
    Blocks of consecutive days preserve short-term structure.
    """
    arr = returns.dropna().to_numpy()
    n = len(arr)
    if n < block_size * 2:
        return {"sharpe": float("nan"), "p05": float("nan"), "p50": float("nan"),
                "p95": float("nan"), "n": n, "note": "insufficient data"}

    rng = np.random.default_rng(seed)
    n_blocks = n // block_size
    sharpes = np.empty(n_resamples, dtype=float)

    for i in range(n_resamples):
        starts = rng.integers(0, n - block_size, size=n_blocks)
        sample = np.concatenate([arr[s : s + block_size] for s in starts])
        std = sample.std()
        sharpes[i] = (sample.mean() / std * np.sqrt(periods_per_year)) if std > 0 else 0.0

    return {
        "sharpe_point": float(arr.mean() / arr.std() * np.sqrt(periods_per_year)) if arr.std() > 0 else 0.0,
        "p05": float(np.percentile(sharpes, 5)),
        "p50": float(np.percentile(sharpes, 50)),
        "p95": float(np.percentile(sharpes, 95)),
        "pct_above_zero": float((sharpes > 0).mean()),
        "n": n,
        "n_resamples": n_resamples,
        "block_size": block_size,
    }


def _binom_two_sided_p(k: int, n: int, p: float = 0.5) -> float:
    """Two-sided binomial p-value. Manual (no scipy dependency)."""
    if n == 0:
        return 1.0
    pmf_obs = comb(n, k) * (p ** k) * ((1 - p) ** (n - k))
    total = 0.0
    for i in range(n + 1):
        pmf = comb(n, i) * (p ** i) * ((1 - p) ** (n - i))
        if pmf <= pmf_obs + 1e-12:
            total += pmf
    return min(total, 1.0)


def trade_sign_test(trades: pd.DataFrame) -> dict:
    """Binomial test on trade win rate under H0: p(win) = 0.5."""
    if len(trades) == 0:
        return {"n_trades": 0, "wins": 0, "win_rate": float("nan"), "p_value": float("nan")}
    wins = int((trades["pnl"] > 0).sum())
    n = len(trades)
    p = _binom_two_sided_p(wins, n)
    return {
        "n_trades": n,
        "wins": wins,
        "win_rate": wins / n,
        "p_value": p,
    }


def format_bootstrap_output(bs: dict, sign: dict) -> str:
    lines = [
        f"  block bootstrap ({bs.get('n_resamples', 0)} resamples, block={bs.get('block_size', 0)}):",
        f"    Sharpe point:      {bs.get('sharpe_point', float('nan')):+.3f}",
        f"    5th  percentile:   {bs.get('p05', float('nan')):+.3f}",
        f"    50th percentile:   {bs.get('p50', float('nan')):+.3f}",
        f"    95th percentile:   {bs.get('p95', float('nan')):+.3f}",
        f"    P(Sharpe > 0):     {bs.get('pct_above_zero', float('nan')):.1%}",
        f"  trade sign test:",
        f"    trades:            {sign['n_trades']}",
        f"    wins:              {sign['wins']}",
        f"    win rate:          {sign['win_rate']:.1%}" if sign['n_trades'] else "",
        f"    p-value (H0: 50%): {sign['p_value']:.3f}" if sign['n_trades'] else "",
    ]
    if sign['n_trades'] and sign['p_value'] > 0.10:
        lines.append("    VERDICT: cannot reject null. Win rate indistinguishable from chance.")
    return "\n".join(line for line in lines if line)
    
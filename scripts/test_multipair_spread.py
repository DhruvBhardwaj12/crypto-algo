"""Option 3: Multi-pair spread mean reversion.

Select the top N most correlated pairs from the universe, run pair-spread
mean reversion on each, combine into one portfolio. Tests whether
diversification across pairs lifts Sharpe.

Run:
    uv run python scripts/test_multipair_spread.py
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from loguru import logger

from crypto_algo.backtesting.portfolio_simulator import run_portfolio_backtest
from crypto_algo.data.loaders import load_parquet
from crypto_algo.strategies.pair_spread import pair_spread_weights

START = "2025-01-01"
END = "2026-09-23"
INITIAL_EQUITY = 10_000.0
PERIODS_PER_YEAR = 6 * 365
COST_BPS = 5.0

# Config
N_PAIRS = 6          # top-N pairs by correlation
LOOKBACK = 42
ENTRY_Z = 2.0
EXIT_Z = 0.5
MIN_CORR = 0.75      # only consider pairs above this correlation


class _Cost:
    def __init__(self, bps: float):
        self.cost_bps = bps

    @property
    def buy_cost_bps(self) -> float:
        return self.cost_bps

    @property
    def sell_cost_bps(self) -> float:
        return self.cost_bps

    @property
    def round_trip_bps(self) -> float:
        return 2 * self.cost_bps

    def cost_fraction(self, side: str) -> float:
        return self.cost_bps / 10_000.0


def _load_all(symbols: list[str]) -> pd.DataFrame:
    series = {}
    for symbol in symbols:
        path = Path(f"data/raw/binance/futures/{symbol}/4h/{START}_{END}.parquet")
        if not path.exists():
            continue
        df = load_parquet(path).sort_values("open_time").set_index("open_time")
        s = df["close"].rename(symbol)
        if len(s) > 100:
            series[symbol] = s
    wide = pd.DataFrame(series).sort_index()
    non_nan = wide.notna().sum(axis=1)
    valid_start = non_nan[non_nan >= 6].index.min()
    return wide.loc[wide.index >= valid_start]


def _select_pairs(prices: pd.DataFrame, n: int) -> list[tuple[str, str]]:
    """Top n pairs by correlation of log returns."""
    log_rets = np.log(prices).diff().dropna()
    corr = log_rets.corr()

    candidates = []
    for a, b in combinations(prices.columns, 2):
        c = corr.loc[a, b]
        if not np.isnan(c) and c >= MIN_CORR:
            candidates.append((a, b, float(c)))

    candidates.sort(key=lambda x: -x[2])
    top = candidates[:n]
    logger.info("Top {} pairs by correlation:", n)
    for a, b, c in top:
        logger.info("  {} / {}: corr={:.3f}", a, b, c)
    return [(a, b) for a, b, _ in top]


def _pair_weights_to_wide(
    prices: pd.DataFrame, pair: tuple[str, str],
) -> pd.DataFrame:
    a, b = pair
    w_narrow = pair_spread_weights(
        prices[a], prices[b],
        lookback=LOOKBACK, entry_z=ENTRY_Z, exit_z=EXIT_Z,
        leg_weight=0.5,
        symbol_a=a,
        symbol_b=b,
    )
    w_wide = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    w_wide[a] = w_narrow[a].reindex(prices.index).fillna(0.0)
    w_wide[b] = w_narrow[b].reindex(prices.index).fillna(0.0)
    return w_wide



def _metrics(equity: pd.Series) -> dict:
    rets = equity.pct_change().dropna()
    total = float(equity.iloc[-1] / equity.iloc[0] - 1.0)
    years = len(equity) / PERIODS_PER_YEAR
    cagr = float((equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1) if years > 0 else float("nan")
    std = float(rets.std())
    sharpe = float(rets.mean() / std * np.sqrt(PERIODS_PER_YEAR)) if std > 0 else float("nan")
    peak = equity.cummax()
    dd = float((equity / peak - 1.0).min())
    return {"total_return": total, "cagr": cagr, "sharpe": sharpe, "max_dd": dd}


def main() -> None:
    with open("config/settings.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)["data"]
    prices = _load_all(cfg["symbols"])

    print(f"\n{'=' * 88}")
    print(f"  Multi-Pair Spread Mean Reversion — {START} to {END}")
    print(f"  {prices.shape[1]} symbols, {prices.shape[0]} bars")
    print(f"{'=' * 88}")

    pairs = _select_pairs(prices, N_PAIRS)
    if not pairs:
        print("No pairs met correlation threshold.")
        return

    print(f"\n  Selected {len(pairs)} pairs: {pairs}")

    # Individual pair results
    print(f"\n  Per-pair results (cost {COST_BPS} bps/side):")
    print(f"  {'pair':>22s}  {'TotRet':>10s}  {'Sharpe':>8s}  {'MaxDD':>9s}")
    pair_weights_list = []
    for a, b in pairs:
        w = _pair_weights_to_wide(prices, (a, b))
        pair_weights_list.append(w)
        r = run_portfolio_backtest(
            prices=prices, weights=w, cost_model=_Cost(COST_BPS),
            initial_equity=INITIAL_EQUITY, periods_per_year=PERIODS_PER_YEAR,
        )
        m = _metrics(r.equity)
        print(f"  {a.replace('USDT','')+'/'+b.replace('USDT',''):>22s}  "
              f"{m['total_return']:>+9.1%}  {m['sharpe']:>+8.2f}  {m['max_dd']:>+9.2%}")

    # Combined equal-weight across all pairs
    combined = sum(pair_weights_list) / len(pair_weights_list)
    r_comb = run_portfolio_backtest(
        prices=prices, weights=combined, cost_model=_Cost(COST_BPS),
        initial_equity=INITIAL_EQUITY, periods_per_year=PERIODS_PER_YEAR,
    )
    m_comb = _metrics(r_comb.equity)

    print(f"\n  Equal-weight portfolio of {len(pairs)} pairs:")
    print(f"    Total return: {m_comb['total_return']:>+10.1%}")
    print(f"    CAGR:         {m_comb['cagr']:>+10.1%}")
    print(f"    Sharpe:       {m_comb['sharpe']:>+10.2f}")
    print(f"    Max drawdown: {m_comb['max_dd']:>+10.2%}")

    # Cost sensitivity
    print(f"\n  Cost sensitivity on portfolio:")
    print(f"  {'bps/side':>9s}  {'Sharpe':>8s}  {'TotRet':>10s}  {'MaxDD':>9s}")
    for bps in [5.0, 10.0, 15.0, 25.0]:
        r = run_portfolio_backtest(
            prices=prices, weights=combined, cost_model=_Cost(bps),
            initial_equity=INITIAL_EQUITY, periods_per_year=PERIODS_PER_YEAR,
        )
        m = _metrics(r.equity)
        print(f"  {bps:>9.1f}  {m['sharpe']:>+8.2f}  {m['total_return']:>+9.1%}  {m['max_dd']:>+9.2%}")


if __name__ == "__main__":
    main()

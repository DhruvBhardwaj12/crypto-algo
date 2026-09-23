"""Option 2: Combined portfolio of CS Reversion + Pair Spread.

Run both strategies on recent data, compute correlation, test 50/50
combined portfolio.

Run:
    uv run python scripts/test_combined_reversion.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from loguru import logger

from crypto_algo.backtesting.portfolio_simulator import run_portfolio_backtest
from crypto_algo.data.loaders import load_parquet
from crypto_algo.strategies.cross_sectional_reversion import (
    cross_sectional_reversion_weights,
)
from crypto_algo.strategies.pair_spread import pair_spread_weights

START = "2025-01-01"
END = "2026-09-23"
INITIAL_EQUITY = 10_000.0
PERIODS_PER_YEAR = 6 * 365
COST_BPS = 5.0

CS_PARAMS = {"lookback": 42, "n_long": 3, "n_short": 3, "rebalance": 6}
PAIR_PARAMS = {"lookback": 42, "entry_z": 2.0, "exit_z": 0.5}


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


def _fmt(v: float) -> str:
    if v != v:
        return "n/a"
    return f"{v:+,.3f}" if abs(v) < 10 else f"{v:+,.1f}"


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
    symbols = cfg["symbols"]

    prices = _load_all(symbols)
    print(f"\n{'=' * 88}")
    print(f"  Combined Reversion Portfolio — {START} to {END}")
    print(f"  {prices.shape[1]} symbols, {prices.shape[0]} bars")
    print(f"{'=' * 88}")

    # --- Strategy 1: CS Reversion ---
    cs_weights = cross_sectional_reversion_weights(
        prices, lookback=CS_PARAMS["lookback"],
        n_long=CS_PARAMS["n_long"], n_short=CS_PARAMS["n_short"],
        rebalance_every=CS_PARAMS["rebalance"],
    )
    r_cs = run_portfolio_backtest(
        prices=prices, weights=cs_weights, cost_model=_Cost(COST_BPS),
        initial_equity=INITIAL_EQUITY, periods_per_year=PERIODS_PER_YEAR,
    )

    # --- Strategy 2: Pair Spread (BTC/ETH only, mapped into wide weights) ---
    pair_weights_narrow = pair_spread_weights(
        prices["BTCUSDT"], prices["ETHUSDT"],
        lookback=PAIR_PARAMS["lookback"],
        entry_z=PAIR_PARAMS["entry_z"],
        exit_z=PAIR_PARAMS["exit_z"],
        leg_weight=0.5,
    )
    pair_weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    for col in ["BTCUSDT", "ETHUSDT"]:
        if col in pair_weights.columns:
            pair_weights[col] = pair_weights_narrow[col].reindex(prices.index).fillna(0.0)
    r_pair = run_portfolio_backtest(
        prices=prices, weights=pair_weights, cost_model=_Cost(COST_BPS),
        initial_equity=INITIAL_EQUITY, periods_per_year=PERIODS_PER_YEAR,
    )

    # --- Combined 50/50 ---
    # Combined weight matrix: average the two strategies' weights.
    combined_weights = 0.5 * cs_weights + 0.5 * pair_weights
    r_combined = run_portfolio_backtest(
        prices=prices, weights=combined_weights, cost_model=_Cost(COST_BPS),
        initial_equity=INITIAL_EQUITY, periods_per_year=PERIODS_PER_YEAR,
    )

    # --- Report ---
    m_cs = _metrics(r_cs.equity)
    m_pair = _metrics(r_pair.equity)
    m_comb = _metrics(r_combined.equity)

    print(f"\n  {'strategy':>22s}  {'TotRet':>10s}  {'CAGR':>8s}  {'Sharpe':>8s}  {'MaxDD':>9s}")
    print(f"  {'CS Reversion':>22s}  {m_cs['total_return']:>+9.1%}  "
          f"{m_cs['cagr']:>+7.1%}  {m_cs['sharpe']:>+8.2f}  {m_cs['max_dd']:>+9.2%}")
    print(f"  {'Pair Spread (BTC/ETH)':>22s}  {m_pair['total_return']:>+9.1%}  "
          f"{m_pair['cagr']:>+7.1%}  {m_pair['sharpe']:>+8.2f}  {m_pair['max_dd']:>+9.2%}")
    print(f"  {'50/50 Combined':>22s}  {m_comb['total_return']:>+9.1%}  "
          f"{m_comb['cagr']:>+7.1%}  {m_comb['sharpe']:>+8.2f}  {m_comb['max_dd']:>+9.2%}")

    # --- Correlation of daily returns ---
    r_cs_daily = r_cs.equity.pct_change().resample("1D").apply(lambda r: (1 + r).prod() - 1).dropna()
    r_pair_daily = r_pair.equity.pct_change().resample("1D").apply(lambda r: (1 + r).prod() - 1).dropna()
    common = pd.concat({"cs": r_cs_daily, "pair": r_pair_daily}, axis=1).dropna()
    corr = common.corr().iloc[0, 1]
    print(f"\n  Daily return correlation (CS Reversion vs Pair Spread): {corr:+.3f}")

    # --- Cost sensitivity on combined ---
    print(f"\n  Cost sensitivity on 50/50 combined:")
    print(f"  {'bps/side':>9s}  {'Sharpe':>8s}  {'TotRet':>10s}  {'MaxDD':>9s}")
    for bps in [5.0, 10.0, 15.0, 25.0]:
        r = run_portfolio_backtest(
            prices=prices, weights=combined_weights, cost_model=_Cost(bps),
            initial_equity=INITIAL_EQUITY, periods_per_year=PERIODS_PER_YEAR,
        )
        m = _metrics(r.equity)
        print(f"  {bps:>9.1f}  {m['sharpe']:>+8.2f}  {m['total_return']:>+9.1%}  {m['max_dd']:>+9.2%}")


if __name__ == "__main__":
    main()
    
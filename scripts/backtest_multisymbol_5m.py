"""Multi-symbol 5m RSI(2) mean reversion — does diversification help?

Each symbol trades independently. Equal-weight portfolio equity is the
average of the 8 symbol equity curves.

Run:
    uv run python scripts/backtest_multisymbol_5m.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from crypto_algo.backtesting.simulator import run_backtest
from crypto_algo.data.loaders import load_parquet
from crypto_algo.strategies.rsi2_intraday import rsi2_intraday_target

START = "2024-07-01"
END = "2025-01-01"
SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT",
    "XRPUSDT", "DOGEUSDT", "ADAUSDT", "AVAXUSDT",
]
INITIAL_EQUITY = 10_000.0
FUTURES_TAKER_BPS = 5.0
PERIODS_PER_YEAR = 12 * 24 * 365


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


COST = _Cost(FUTURES_TAKER_BPS)


def _load(symbol: str) -> pd.DataFrame:
    return load_parquet(
        Path(f"data/raw/binance/futures/{symbol}/5m/{START}_{END}.parquet")
    )


def _run_symbol(symbol: str) -> tuple[pd.Series, dict]:
    df = _load(symbol)
    target = rsi2_intraday_target(df).reset_index(drop=True)
    target.index = df.index

    result = run_backtest(df, target, COST, INITIAL_EQUITY, PERIODS_PER_YEAR)

    pos = result.position
    n_entries = int(((pos.diff() == 1.0).sum()))
    years = (df["open_time"].iloc[-1] - df["open_time"].iloc[0]).days / 365.25

    info = {
        "symbol": symbol,
        "final_equity": result.final_equity,
        "total_return": result.metrics.get("total_return", float("nan")),
        "sharpe": result.metrics.get("sharpe", float("nan")),
        "max_dd": result.metrics.get("max_drawdown", float("nan")),
        "win_rate": result.metrics.get("win_rate", float("nan")),
        "profit_factor": result.metrics.get("profit_factor", float("nan")),
        "n_entries": n_entries,
        "trades_per_day": n_entries / max(years * 365.25, 1),
    }
    return result.equity, info


def main() -> None:
    print(f"\n{'=' * 90}")
    print(f"  Multi-symbol 5m RSI(2)  —  {START} to {END}  —  {len(SYMBOLS)} symbols")
    print(f"{'=' * 90}")
    print(f"Cost: {FUTURES_TAKER_BPS:.1f} bps/side, {2 * FUTURES_TAKER_BPS:.1f} bps round trip "
          f"(platform fee only, no GST, no slippage)")

    # Run each symbol.
    equities = {}
    infos = []
    for symbol in SYMBOLS:
        try:
            eq, info = _run_symbol(symbol)
            equities[symbol] = eq
            infos.append(info)
        except Exception as exc:  # noqa: BLE001
            logger.warning("{} failed: {}", symbol, exc)

    if not equities:
        print("No symbols ran successfully.")
        return

    # Per-symbol summary table.
    print(f"\n{'Symbol':10s} {'Final$':>10s} {'Return':>10s} {'Sharpe':>9s} "
          f"{'MaxDD':>9s} {'Win%':>7s} {'PF':>7s} {'Trades':>8s} {'Trd/day':>8s}")
    print("-" * 90)
    for i in infos:
        print(
            f"{i['symbol']:10s} {i['final_equity']:>10.2f} "
            f"{i['total_return']:>+9.2%} {i['sharpe']:>9.2f} "
            f"{i['max_dd']:>+9.2%} {i['win_rate']:>7.2%} "
            f"{i['profit_factor']:>7.3f} {i['n_entries']:>8d} "
            f"{i['trades_per_day']:>8.1f}"
        )

    # Build equal-weight portfolio equity.
    # Align on common time index. Use the intersection of all series.
    common_idx = None
    for eq in equities.values():
        common_idx = eq.index if common_idx is None else common_idx.intersection(eq.index)

    # Normalize each equity curve to 1.0 at start, then average.
    normalized = []
    for symbol, eq in equities.items():
        eq_common = eq.reindex(common_idx).ffill().dropna()
        if len(eq_common) < 100:
            continue
        normalized.append(eq_common / eq_common.iloc[0])

    if not normalized:
        print("Could not align equity curves.")
        return

    norm_df = pd.concat(normalized, axis=1)
    norm_df.columns = [s for s in equities.keys()]

    # Portfolio = equal-weight average of normalized curves.
    portfolio = norm_df.mean(axis=1) * INITIAL_EQUITY

    # Portfolio metrics.
    rets = portfolio.pct_change().dropna()
    std = float(rets.std())
    mean = float(rets.mean())
    ann = np.sqrt(PERIODS_PER_YEAR)
    portfolio_sharpe = mean / std * ann if std > 0 else float("nan")
    total_return = float(portfolio.iloc[-1] / portfolio.iloc[0] - 1.0)
    peak = portfolio.cummax()
    dd = portfolio / peak - 1.0
    max_dd = float(dd.min())

    # Correlation matrix of returns across symbols.
    print(f"\n{'=' * 90}")
    print(f"  Equal-weight portfolio of {len(normalized)} symbols")
    print(f"{'=' * 90}")
    print(f"  Final equity:    {portfolio.iloc[-1]:>12.2f}")
    print(f"  Total return:    {total_return:>+12.2%}")
    print(f"  Sharpe:          {portfolio_sharpe:>12.2f}")
    print(f"  Max drawdown:    {max_dd:>+12.2%}")

    # Compare: average of individual Sharpes.
    avg_sharpe = float(np.mean([i["sharpe"] for i in infos]))
    print(f"\n  Average single-symbol Sharpe:  {avg_sharpe:>+8.2f}")
    print(f"  Portfolio Sharpe:               {portfolio_sharpe:>+8.2f}")
    print(f"  Diversification benefit:       {portfolio_sharpe - avg_sharpe:>+8.2f}")

    # Return correlation matrix.
    rets_df = norm_df.pct_change().dropna()
    corr = rets_df.corr()
    print(f"\n  Return correlation matrix (5m bars):")
    print(corr.round(2).to_string())

    avg_corr = float(
        corr.values[np.triu_indices_from(corr, k=1)].mean()
    )
    print(f"\n  Average pairwise correlation: {avg_corr:+.2f}")

    print(f"\n  Interpretation:")
    if portfolio_sharpe > avg_sharpe + 0.5:
        print("    Diversification meaningfully improves Sharpe.")
    elif portfolio_sharpe > avg_sharpe:
        print("    Diversification slightly improves Sharpe.")
    else:
        print("    Diversification does NOT improve Sharpe.")
    if portfolio_sharpe < 0:
        print("    But the portfolio is still a losing strategy.")
        print("    Diversification reduces variance. It does NOT change expectancy.")


if __name__ == "__main__":
    main()
    
"""Orchestrate the full validation suite on the MA crossover (EXP-002).

Run:
    uv run python scripts/validate_ma.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from loguru import logger

from crypto_algo.backtesting.costs import CostModel
from crypto_algo.data.loaders import load_parquet
from crypto_algo.validation.bootstrap import (
    block_bootstrap_sharpe,
    format_bootstrap_output,
    trade_sign_test,
)
from crypto_algo.validation.common import ma_strategy, run_strategy
from crypto_algo.validation.parameter_sweep import build_grid, parameter_sweep, summarize_sweep
from crypto_algo.validation.regime import (
    calendar_year_breakdown,
    format_regime_output,
    volatility_regime_breakdown,
)
from crypto_algo.validation.walk_forward import walk_forward

START = "2020-01-01"
END = "2025-01-01"
SYMBOLS = ["BTCUSDT", "ETHUSDT"]
INITIAL_EQUITY = 10_000.0
CHOSEN = {"fast": 20, "slow": 100}
COST = CostModel()
PERIODS = 365


def _load(symbol: str) -> pd.DataFrame:
    return load_parquet(Path(f"data/raw/binance/{symbol}/1d/{START}_{END}.parquet"))


def _benchmark_returns(df: pd.DataFrame) -> pd.Series:
    """Daily returns of buy & hold on the underlying close."""
    rets = df.set_index("open_time")["close"].pct_change().fillna(0.0)
    return rets


def _plot_sweep(sweep: pd.DataFrame, symbol: str, out: Path) -> None:
    """Heatmap of Sharpe over (fast, slow)."""
    if sweep.empty:
        return
    pivot = sweep.pivot_table(index="fast", columns="slow", values="sharpe")
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn", origin="lower")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel("slow_window")
    ax.set_ylabel("fast_window")
    ax.set_title(f"Sharpe over parameter grid — {symbol}")
    fig.colorbar(im, ax=ax, label="Sharpe")
    plt.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=120)
    plt.close(fig)


def main() -> None:
    grid = build_grid(fast_range=[5, 10, 15, 20, 25, 30, 40, 50], slow_range=[50, 75, 100, 125, 150, 200])

    for symbol in SYMBOLS:
        df = _load(symbol)
        print(f"\n{'=' * 72}")
        print(f"  {symbol}  —  daily data  —  {df['open_time'].iloc[0].date()} to {df['open_time'].iloc[-1].date()}")
        print(f"{'=' * 72}")

        # ---- 1. Parameter stability ----------------------------------------
        print("\n[1] Parameter sweep")
        sweep = parameter_sweep(df, ma_strategy, grid, COST, INITIAL_EQUITY, PERIODS)
        print(summarize_sweep(sweep, CHOSEN))
        _plot_sweep(sweep, symbol, Path(f"research/validation/{symbol}_sweep.png"))

        # ---- 2. Walk-forward ----------------------------------------------
        print("\n[2] Walk-forward (anchored, expanding window)")
        wf = walk_forward(
            df, ma_strategy, grid, n_windows=5,
            cost_model=COST, initial_equity=INITIAL_EQUITY,
            periods_per_year=PERIODS, warmup_bars=110,
        )
        print(wf["summary"])

        # ---- 3. Regime analysis -------------------------------------------
        print("\n[3] Regime breakdown")
        chosen_result = run_strategy(df, ma_strategy, CHOSEN, COST, INITIAL_EQUITY, PERIODS)
        bench = _benchmark_returns(df)
        bench.index = pd.to_datetime(bench.index, utc=True)
        cal = calendar_year_breakdown(chosen_result, bench, PERIODS)
        vol = volatility_regime_breakdown(chosen_result, bench, PERIODS)
        print(format_regime_output(cal, vol))

        # ---- 4. Bootstrap -------------------------------------------------
        print("\n[4] Statistical significance")
        bs = block_bootstrap_sharpe(chosen_result.returns, periods_per_year=PERIODS)
        sign = trade_sign_test(chosen_result.trades)
        print(format_bootstrap_output(bs, sign))

    logger.success("Validation suite complete.")


if __name__ == "__main__":
    main()
    
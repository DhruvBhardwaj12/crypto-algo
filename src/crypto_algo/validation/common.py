"""Shared helpers for the validation framework.

A strategy is a function with the signature:

    strategy_fn(close: pd.Series, params: dict) -> pd.Series

returning a target position in [0, 1] for each bar. The validation modules
call strategies via run_strategy(), which glues them to the backtester.
"""

from __future__ import annotations

from typing import Protocol

import pandas as pd

from crypto_algo.backtesting.costs import CostModel
from crypto_algo.backtesting.simulator import BacktestResult, run_backtest


class StrategyFn(Protocol):
    def __call__(self, close: pd.Series, params: dict) -> pd.Series: ...


def run_strategy(
    df: pd.DataFrame,
    strategy_fn: StrategyFn,
    params: dict,
    cost_model: CostModel,
    initial_equity: float,
    periods_per_year: int,
) -> BacktestResult:
    """Apply strategy to df, run the backtester, return the result."""
    target = strategy_fn(df["close"], params).reset_index(drop=True)
    target.index = df.index
    return run_backtest(df, target, cost_model, initial_equity, periods_per_year)


def ma_strategy(close: pd.Series, params: dict) -> pd.Series:
    """Adapter for the MA crossover. Params: {'fast': int, 'slow': int}."""
    from crypto_algo.strategies.ma_crossover import ma_crossover_target

    return ma_crossover_target(
        close, fast_window=params["fast"], slow_window=params["slow"]
    )


    
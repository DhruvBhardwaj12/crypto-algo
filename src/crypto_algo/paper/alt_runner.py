"""Generic paper trader runner for TSMOM and 5m RSI(2).

Shares the state/feed/executor infrastructure with the existing Donchian
runner. The only difference is the signal function, dispatched by the
`strategy.name` field in the config.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
from loguru import logger

from crypto_algo.paper.executor import mark_to_market, simulate_fill
from crypto_algo.paper.feed import fetch_recent_klines
from crypto_algo.paper.state import PaperState
from crypto_algo.strategies.rsi2_intraday import rsi2_intraday_target
from crypto_algo.strategies.tsmom import tsmom_target


def _append_log(log_path: Path, record: dict) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a") as f:
        f.write(json.dumps(record, default=str) + "\n")


def _tsmom_signal(df: pd.DataFrame, params: dict) -> int:
    target = tsmom_target(df["close"], lookback_n=int(params["lookback"]))
    return int(target.iloc[-1])


def _rsi2_5m_signal(df: pd.DataFrame, params: dict) -> int:
    target = rsi2_intraday_target(
        df,
        rsi_period=int(params["rsi_period"]),
        entry_threshold=float(params["entry_threshold"]),
        exit_threshold=float(params["exit_threshold"]),
    )
    return int(target.iloc[-1])


_SIGNAL_DISPATCH = {
    "tsmom": _tsmom_signal,
    "rsi2_5m": _rsi2_5m_signal,
}


def run_once(
    cfg: dict,
    state: PaperState,
    state_path: Path,
    log_path: Path,
) -> None:
    symbol = cfg["strategy"]["symbol"]
    interval = cfg["strategy"]["interval"]
    strategy_name = cfg["strategy"]["name"]
    params = cfg["params"]
    cost_bps = cfg["execution"]["cost_bps_per_side"]

    if strategy_name not in _SIGNAL_DISPATCH:
        raise ValueError(f"Unknown strategy: {strategy_name}")
    signal_fn = _SIGNAL_DISPATCH[strategy_name]

    df = fetch_recent_klines(symbol, interval, limit=500)

    now = pd.Timestamp.now(tz="UTC")
    closed = df[df["close_time"] < now].reset_index(drop=True)
    if len(closed) < 60:
        logger.warning("Insufficient closed history ({} bars).", len(closed))
        return

    last_closed = closed.iloc[-1]
    last_closed_open = str(last_closed["open_time"])

    if state.last_bar_time == last_closed_open:
        return

    # Execute pending signal at this bar's open.
    if state.pending_signal is not None:
        outcome = simulate_fill(
            equity=state.equity,
            position=state.position,
            units=state.units,
            entry_price=state.entry_price,
            entry_time=state.entry_time,
            entry_equity=state.entry_equity,
            target_position=int(state.pending_signal),
            fill_price=float(last_closed["open"]),
            fill_time=str(last_closed["open_time"]),
            cost_bps_per_side=cost_bps,
        )
        state.equity = outcome.equity
        state.position = outcome.position
        state.units = outcome.units
        state.entry_price = outcome.entry_price
        state.entry_time = outcome.entry_time
        state.entry_equity = outcome.entry_equity
        if outcome.trade_pnl is not None:
            state.n_trades += 1
            if outcome.trade_won:
                state.n_wins += 1
            else:
                state.n_losses += 1
        state.pending_signal = None
        state.pending_signal_bar = None

    # Mark-to-market with this bar's close.
    mtm = mark_to_market(
        units=state.units,
        position=state.position,
        equity_when_flat=state.equity,
        close_price=float(last_closed["close"]),
    )
    state.equity = mtm

    # New signal for this bar.
    new_signal = signal_fn(closed, params)
    if new_signal != state.position:
        state.pending_signal = new_signal
        state.pending_signal_bar = last_closed_open

    state.last_bar_time = last_closed_open
    state.save(state_path)

    _append_log(log_path, {
        "ts_utc": str(now),
        "strategy": strategy_name,
        "symbol": symbol,
        "bar_open_time": last_closed_open,
        "close": float(last_closed["close"]),
        "signal_for_next": new_signal,
        "current_position": state.position,
        "equity": state.equity,
        "n_trades": state.n_trades,
        "n_wins": state.n_wins,
        "n_losses": state.n_losses,
    })

    logger.info(
        "{} {} bar={} close={:.4f} pos={} eq={:.2f} trades={}",
        strategy_name, symbol, last_closed_open,
        float(last_closed["close"]), state.position,
        state.equity, state.n_trades,
    )


def run_forever(cfg: dict) -> None:
    state_path = Path(cfg["paths"]["state_file"])
    log_path = Path(cfg["paths"]["log_file"])

    state = PaperState.load_or_init(
        state_path,
        symbol=cfg["strategy"]["symbol"],
        strategy_name=cfg["strategy"]["name"],
        initial_equity=cfg["execution"]["initial_equity"],
    )

    logger.info(
        "Paper trader started: {} {} (equity={:.2f}, trades={})",
        cfg["strategy"]["name"], cfg["strategy"]["symbol"],
        state.equity, state.n_trades,
    )

    interval = cfg["poll"]["interval_seconds"]
    try:
        while True:
            try:
                run_once(cfg, state, state_path, log_path)
            except Exception as exc:  # noqa: BLE001
                logger.exception("Iteration failed: {}", exc)
            time.sleep(interval)
    except KeyboardInterrupt:
        logger.info("Paper trader stopped by user. State saved.")
        
"""Paper trading loop for the Donchian + RSI strategy."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from loguru import logger

from crypto_algo.paper.executor import mark_to_market, simulate_fill
from crypto_algo.paper.feed import fetch_recent_klines
from crypto_algo.paper.state import PaperState
from crypto_algo.strategies.donchian_rsi import donchian_rsi_target, rsi, tag_with_4h_regime


def _append_log(log_path: Path, record: dict) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a") as f:
        f.write(json.dumps(record, default=str) + "\n")


def _compute_signal_on_history(
    df_1h: pd.DataFrame,
    df_4h: pd.DataFrame,
    params: dict,
) -> int:
    """Compute the strategy's target at the last closed bar of df_1h."""
    rsi_4h = tag_with_4h_regime(df_1h, df_4h, rsi_period=params["rsi_4h_period"])
    target = donchian_rsi_target(
        df_1h,
        rsi_4h,
        donchian_entry_lookback=params["dc"],
        donchian_exit_lookback=params["dc_exit"],
        rsi_1h_period=params["rsi_1h_period"],
        rsi_1h_entry=params["rsi_1h_entry"],
        rsi_4h_entry=params["rsi_4h_entry"],
        rsi_4h_exit=params["rsi_4h_exit"],
    )
    return int(target.iloc[-1])


def run_once(
    cfg: dict,
    state: PaperState,
    state_path: Path,
    log_path: Path,
) -> None:
    """One iteration: fetch, process new closed bars, update state, log."""
    symbol = cfg["strategy"]["symbol"]
    entry_iv = cfg["strategy"]["entry_interval"]
    regime_iv = cfg["strategy"]["regime_interval"]
    params = cfg["params"]
    cost_bps = cfg["execution"]["cost_bps_per_side"]

    df_1h = fetch_recent_klines(symbol, entry_iv, limit=500)
    df_4h = fetch_recent_klines(symbol, regime_iv, limit=300)

    # Last CLOSED 1H bar = the one whose close_time < now.
    now = pd.Timestamp.now(tz="UTC")
    closed = df_1h[df_1h["close_time"] < now].reset_index(drop=True)
    if len(closed) < 200:
        logger.warning("Insufficient closed history ({} bars). Skipping.", len(closed))
        return

    last_closed = closed.iloc[-1]
    last_closed_open = str(last_closed["open_time"])

    # If nothing new, do nothing.
    if state.last_bar_time == last_closed_open:
        return

    # If we have a pending signal from the previous bar, execute at this bar's open.
    if state.pending_signal is not None and state.pending_signal_bar is not None:
        # Find the bar whose open_time equals last_closed_open — the pending signal
        # executes at the OPEN of the first bar after pending_signal_bar.
        execute_bar = closed.iloc[-1]
        outcome = simulate_fill(
            equity=state.equity,
            position=state.position,
            units=state.units,
            entry_price=state.entry_price,
            entry_time=state.entry_time,
            entry_equity=state.entry_equity,
            target_position=int(state.pending_signal),
            fill_price=float(execute_bar["open"]),
            fill_time=str(execute_bar["open_time"]),
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

    # Compute new signal for this bar.
    new_signal = _compute_signal_on_history(closed, df_4h, params)

    # Record new pending signal (execute on next closed bar's open).
    if new_signal != state.position:
        state.pending_signal = new_signal
        state.pending_signal_bar = last_closed_open

    state.last_bar_time = last_closed_open
    state.save(state_path)

    _append_log(log_path, {
        "ts_utc": str(now),
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
        "donchian_rsi {} bar={} close={:.2f} pos={} eq={:.2f} trades={}",
        cfg["strategy"]["symbol"], last_closed_open,
        float(last_closed["close"]),
        state.position,
        state.equity,
        state.n_trades,
    )
    
    logger.info(
        "bar={} close={:.2f} pos={} equity={:.2f} trades={}",
        last_closed_open, float(last_closed["close"]), state.position, state.equity, state.n_trades,
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
        
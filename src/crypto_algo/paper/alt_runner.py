"""Paper trader for TSMOM and 5m RSI(2). Now with risk engine integration."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
from loguru import logger

from crypto_algo.paper.executor import mark_to_market, simulate_fill
from crypto_algo.paper.feed import fetch_recent_klines
from crypto_algo.paper.state import PaperState
from crypto_algo.risk.engine import RiskDecision, RiskEngine
from crypto_algo.strategies.rsi2_intraday import rsi2_intraday_target
from crypto_algo.strategies.tsmom import tsmom_target


def _append_log(log_path: Path, record: dict) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a") as f:
        f.write(json.dumps(record, default=str) + "\n")


def _tsmom_signal(df: pd.DataFrame, params: dict) -> float:
    target = tsmom_target(df["close"], lookback_n=int(params["lookback"]))
    return float(target.iloc[-1])


def _rsi2_5m_signal(df: pd.DataFrame, params: dict) -> float:
    target = rsi2_intraday_target(
        df,
        rsi_period=int(params["rsi_period"]),
        entry_threshold=float(params["entry_threshold"]),
        exit_threshold=float(params["exit_threshold"]),
    )
    return float(target.iloc[-1])

def _donchian_rsi_signal(
    df_1h: pd.DataFrame,
    df_4h: pd.DataFrame,
    params: dict,
) -> float:
    """Donchian + RSI multi-timeframe. Needs both 1H and 4H frames."""
    from crypto_algo.strategies.donchian_rsi import donchian_rsi_target, tag_with_4h_regime

    rsi_4h = tag_with_4h_regime(df_1h, df_4h, rsi_period=int(params["rsi_4h_period"]))
    target = donchian_rsi_target(
        df_1h,
        rsi_4h,
        donchian_entry_lookback=int(params["dc"]),
        donchian_exit_lookback=int(params["dc_exit"]),
        rsi_1h_period=int(params["rsi_1h_period"]),
        rsi_1h_entry=float(params["rsi_1h_entry"]),
        rsi_4h_entry=float(params["rsi_4h_entry"]),
        rsi_4h_exit=float(params["rsi_4h_exit"]),
    )
    return float(target.iloc[-1])



_SIGNAL_DISPATCH = {
    "tsmom": _tsmom_signal,
    "rsi2_5m": _rsi2_5m_signal,
    "donchian_rsi": None,  # handled specially below (needs two timeframes)
}

def _apply_risk(
    engine: RiskEngine,
    strategy_name: str,
    symbol: str,
    equity: float,
    current_units: float,
    fill_price: float,
    proposed_signal: float,
    state: PaperState,
    log_path: Path,
) -> float:
    """Run risk engine. Returns approved target position (may be reduced/rejected)."""
    current_weights = pd.Series({symbol: (current_units * fill_price) / equity if equity > 0 else 0.0})
    proposed_weights = pd.Series({symbol: proposed_signal})

    report = engine.evaluate(equity, current_weights, proposed_weights)

    if report.decision == RiskDecision.REJECT:
        state.n_risk_rejections += 1
        logger.warning("RISK REJECT {} {}: {}", strategy_name, symbol, report.reasons)
        _append_log(log_path, {
            "event": "risk_reject", "reasons": report.reasons,
            "proposed": proposed_signal,
        })
        return 0.0
    if report.decision == RiskDecision.REDUCE:
        state.n_risk_reductions += 1
        reduced = float(report.approved_weights.get(symbol, 0.0))
        logger.info("RISK REDUCE {} {}: {:.2f} -> {:.2f}",
                    strategy_name, symbol, proposed_signal, reduced)
        _append_log(log_path, {
            "event": "risk_reduce", "reasons": report.reasons,
            "proposed": proposed_signal, "approved": reduced,
        })
        return reduced

    return proposed_signal


def run_once(
    cfg: dict,
    state: PaperState,
    state_path: Path,
    log_path: Path,
    engine: RiskEngine,
) -> None:
    symbol = cfg["strategy"]["symbol"]
    interval = cfg["strategy"]["interval"]
    strategy_name = cfg["strategy"]["name"]
    params = cfg["params"]
    cost_bps = cfg["execution"]["cost_bps_per_side"]

    if strategy_name not in _SIGNAL_DISPATCH:
        raise ValueError(f"Unknown strategy: {strategy_name}")

    # Multi-timeframe strategies fetch an additional frame.
    df_4h = None
    if strategy_name == "donchian_rsi":
        df_1h = fetch_recent_klines(symbol, "1h", limit=500)
        df_4h = fetch_recent_klines(symbol, "4h", limit=500)
        df = df_1h
    else:
        df = fetch_recent_klines(symbol, interval, limit=500)

    now = pd.Timestamp.now(tz="UTC")
    closed = df[df["close_time"] < now].reset_index(drop=True)
    if len(closed) < 60:
        return

    last_closed = closed.iloc[-1]
    last_closed_open = str(last_closed["open_time"])

    if state.last_bar_time == last_closed_open:
        return

    # Update risk engine's view of equity and roll the day if needed.
    current_equity = mark_to_market(state.cash, state.units, float(last_closed["close"]))
    engine.update_equity(current_equity)

    today = now.strftime("%Y-%m-%d")
    if state.last_day != today:
        state.day_start_equity = current_equity
        state.last_day = today
        engine.new_day(current_equity)

    state.peak_equity = max(state.peak_equity, current_equity)

    # Execute pending signal.
    if state.pending_signal is not None:
        approved_signal = _apply_risk(
            engine=engine,
            strategy_name=strategy_name,
            symbol=symbol,
            equity=current_equity,
            current_units=state.units,
            fill_price=float(last_closed["open"]),
            proposed_signal=state.pending_signal,
            state=state,
            log_path=log_path,
        )
        outcome = simulate_fill(
            cash=state.cash,
            units=state.units,
            equity=current_equity,
            entry_price=state.entry_price,
            entry_time=state.entry_time,
            entry_equity=state.entry_equity,
            target_position=approved_signal,
            fill_price=float(last_closed["open"]),
            fill_time=str(last_closed["open_time"]),
            cost_bps_per_side=cost_bps,
        )
        state.cash = outcome.cash
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

        # Compute new signal.
    if strategy_name == "donchian_rsi":
        new_signal = _donchian_rsi_signal(closed, df_4h, params)
    else:
        new_signal = _SIGNAL_DISPATCH[strategy_name](closed, params)
    mtm_equity = mark_to_market(state.cash, state.units, float(last_closed["close"]))

    # Compare new signal to current position (as fraction of equity).
    current_pos_frac = (state.units * float(last_closed["close"])) / mtm_equity if mtm_equity > 0 else 0.0
    if abs(new_signal - current_pos_frac) > 1e-6:
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
        "current_pos_frac": current_pos_frac,
        "equity": mtm_equity,
        "cash": state.cash,
        "units": state.units,
        "n_trades": state.n_trades,
        "n_wins": state.n_wins,
        "n_losses": state.n_losses,
        "n_risk_rejections": state.n_risk_rejections,
        "n_risk_reductions": state.n_risk_reductions,
    })

    logger.info(
        "{} {} bar={} close={:.4f} pos={:.2f} eq={:.2f} trades={} rej={}",
        strategy_name, symbol, last_closed_open,
        float(last_closed["close"]), current_pos_frac,
        mtm_equity, state.n_trades, state.n_risk_rejections,
    )


def run_forever(cfg: dict) -> None:
    state_path = Path(cfg["paths"]["state_file"])
    log_path = Path(cfg["paths"]["log_file"])
    risk_cfg = Path(cfg["risk"]["config_file"])

    state = PaperState.load_or_init(
        state_path,
        symbol=cfg["strategy"]["symbol"],
        strategy_name=cfg["strategy"]["name"],
        initial_equity=cfg["execution"]["initial_equity"],
    )
    engine = RiskEngine(risk_cfg)

    logger.info(
        "Paper trader started: {} {} (equity={:.2f}, trades={})",
        cfg["strategy"]["name"], cfg["strategy"]["symbol"],
        state.cash, state.n_trades,
    )

    interval = cfg["poll"]["interval_seconds"]
    try:
        while True:
            try:
                run_once(cfg, state, state_path, log_path, engine)
            except Exception as exc:  # noqa: BLE001
                logger.exception("Iteration failed: {}", exc)
            time.sleep(interval)
    except KeyboardInterrupt:
        logger.info("Paper trader stopped by user. State saved.")

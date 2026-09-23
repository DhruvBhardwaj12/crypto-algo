"""Paper trading dashboard.

Reads all paper trading logs in research/paper_trading/ and prints a
daily summary. Run this before you check email every morning.

Run:
    uv run python scripts/dashboard.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

LOG_DIR = Path("research/paper_trading")
STALE_MINUTES = 15


def _read_log(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _read_state(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        with open(path, "r") as f:
            return json.load(f)
    except json.JSONDecodeError:
        return None


def _fmt_money(v: float) -> str:
    return f"${v:,.2f}"


def _fmt_pct(v: float) -> str:
    return f"{v:+.2%}"


def _fmt_ago(minutes: float) -> str:
    if minutes < 1:
        return "just now"
    if minutes < 60:
        return f"{int(minutes)}m ago"
    if minutes < 60 * 24:
        return f"{minutes / 60:.1f}h ago"
    return f"{minutes / (60 * 24):.1f}d ago"


def _minutes_since(iso_ts: str | None) -> float | None:
    if not iso_ts:
        return None
    try:
        ts = pd.Timestamp(iso_ts)
        if ts.tz is None:
            ts = ts.tz_localize("UTC")
        return (pd.Timestamp.now(tz="UTC") - ts).total_seconds() / 60
    except Exception:
        return None


def _summarize(t: dict) -> dict:
    rows = t["rows"]
    state = t["state"]
    if not rows:
        return {"stem": t["stem"], "empty": True, "state": state}

    last = rows[-1]
    last_bar_time = last.get("bar_open_time") or last.get("ts_utc")

    eq_series = pd.Series(
        [r.get("equity", 0.0) for r in rows],
        index=pd.to_datetime(
            [r.get("bar_open_time", r.get("ts_utc")) for r in rows],
            utc=True, errors="coerce",
        ),
    ).dropna()
    eq_series = eq_series[~eq_series.index.duplicated(keep="last")].sort_index()

    start_eq = state.get("initial_equity") if state else None
    if start_eq is None and len(eq_series) > 0:
        start_eq = eq_series.iloc[0]
    current_eq = eq_series.iloc[-1] if len(eq_series) > 0 else None

    daily_eq = eq_series.resample("1D").last().dropna()
    daily_rets = daily_eq.pct_change().dropna()
    if len(daily_rets) > 5 and daily_rets.std() > 0:
        sharpe = float(daily_rets.mean() / daily_rets.std() * (365 ** 0.5))
    else:
        sharpe = float("nan")

    n_trades = last.get("n_trades", 0)
    n_wins = last.get("n_wins", 0)

    return {
        "stem": t["stem"],
        "empty": False,
        "strategy": last.get("strategy", "unknown"),
        "symbol": last.get("symbol", "unknown"),
        "start_equity": start_eq,
        "current_equity": current_eq,
        "total_return": (current_eq / start_eq - 1.0) if start_eq else float("nan"),
        "sharpe_daily": sharpe,
        "n_bars": len(rows),
        "last_bar_time": last_bar_time,
        "minutes_since": _minutes_since(last_bar_time),
        "position_frac": last.get("current_pos_frac", 0.0),
        "n_trades": n_trades,
        "n_wins": n_wins,
        "n_losses": last.get("n_losses", 0),
        "n_risk_rejections": last.get("n_risk_rejections", 0),
        "n_risk_reductions": last.get("n_risk_reductions", 0),
        "win_rate": (n_wins / n_trades) if n_trades > 0 else float("nan"),
        "state": state,
    }


def _print_trader(s: dict) -> None:
    if s.get("empty"):
        print(f"\n  {s['stem']:28s}  (no log entries yet)")
        return

    ms = s.get("minutes_since")
    stale = f"  [STALE {_fmt_ago(ms)}]" if ms is not None and ms > STALE_MINUTES else ""

    print(f"\n  {s['strategy']:14s} {s['symbol']:10s}  "
          f"eq={_fmt_money(s['current_equity'])}  "
          f"ret={_fmt_pct(s['total_return'])}  "
          f"pos={s['position_frac']:.2f}  "
          f"trades={s['n_trades']}{stale}")

    detail = [f"bars={s['n_bars']}"]
    if s["n_trades"] > 0:
        detail.append(f"W/L={s['n_wins']}/{s['n_losses']}")
        detail.append(f"wr={s['win_rate']:.0%}")
    if s["sharpe_daily"] == s["sharpe_daily"]:
        detail.append(f"sharpe={s['sharpe_daily']:+.2f}")
    if s["n_risk_rejections"] or s["n_risk_reductions"]:
        detail.append(f"risk: {s['n_risk_rejections']}R/{s['n_risk_reductions']}D")

    print(f"    {'  '.join(detail)}")

    st = s.get("state") or {}
    if st:
        print(f"    cash={_fmt_money(st.get('cash', 0))}  "
              f"units={st.get('units', 0):.8f}  "
              f"last_bar={st.get('last_bar_time', 'n/a')}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true", help="show all details")
    args = parser.parse_args()

    if not LOG_DIR.exists():
        print(f"No log directory at {LOG_DIR}")
        return

    log_paths = sorted(LOG_DIR.glob("*_log.jsonl"))
    if not log_paths:
        print(f"No log files found in {LOG_DIR}")
        return

    traders = []
    for log_path in log_paths:
        stem = log_path.stem.replace("_log", "")
        state_path = LOG_DIR / f"{stem}_state.json"
        traders.append({
            "stem": stem,
            "rows": _read_log(log_path),
            "state": _read_state(state_path),
        })

    summaries = [_summarize(t) for t in traders]

    print(f"\n{'=' * 78}")
    print(f"  Paper Trading Dashboard  —  {pd.Timestamp.now(tz='UTC').strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"{'=' * 78}")

    active = [s for s in summaries if not s.get("empty") and s.get("current_equity")]
    if active:
        total_eq = sum(s["current_equity"] for s in active)
        total_start = sum(s["start_equity"] or 0 for s in active)
        total_ret = (total_eq / total_start - 1.0) if total_start > 0 else float("nan")
        n_stale = sum(1 for s in active if (s.get("minutes_since") or 0) > STALE_MINUTES)

        print(f"\n  PORTFOLIO")
        print(f"    Total equity:      {_fmt_money(total_eq)}")
        print(f"    Total return:      {_fmt_pct(total_ret)}")
        print(f"    Active traders:    {len(active)}")
        print(f"    Stale traders:     {n_stale}")

    print(f"\n  TRADERS")
    for s in summaries:
        _print_trader(s)

    print(f"\n  RISK ENGINE")
    total_rej = sum((s.get("n_risk_rejections") or 0) for s in summaries if not s.get("empty"))
    total_red = sum((s.get("n_risk_reductions") or 0) for s in summaries if not s.get("empty"))
    print(f"    Total rejections:  {total_rej}")
    print(f"    Total reductions:  {total_red}")
    print(f"    Kill switch file:  "
          f"{'PRESENT (all trading halted)' if Path('research/KILL_SWITCH').exists() else 'not present'}")

    print()


if __name__ == "__main__":
    main()
    
"""Paper trading status — compact dashboard."""

import json
from pathlib import Path

import pandas as pd

LOG_DIR = Path("research/paper_trading")
STALE_MIN = 15


def read_rows(path):
    if not path.exists():
        return []
    out = []
    for line in open(path, "r"):
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return out


def eq_of(row):
    if "cash" in row and "units" in row and "close" in row:
        return float(row["cash"]) + float(row["units"]) * float(row["close"])
    return float(row.get("equity", 0.0))


def mins_since(ts_str):
    if not ts_str:
        return None
    try:
        ts = pd.Timestamp(ts_str)
        if ts.tz is None:
            ts = ts.tz_localize("UTC")
        return (pd.Timestamp.now(tz="UTC") - ts).total_seconds() / 60
    except Exception:
        return None


def main():
    print("=" * 72)
    print(f"  Paper Trading Status — {pd.Timestamp.now(tz='UTC').strftime('%Y-%m-%d %H:%M UTC')}")
    print("=" * 72)

    total_eq = 0.0
    total_start = 0.0
    n_active = 0
    n_stale = 0

    for log_path in sorted(LOG_DIR.glob("*_log.jsonl")):
        stem = log_path.stem.replace("_log", "")
        state_path = LOG_DIR / f"{stem}_state.json"
        rows = read_rows(log_path)
        state = json.load(open(state_path)) if state_path.exists() else {}

        if not rows:
            print(f"\n  {stem}: no log entries")
            continue

        last = rows[-1]
        curr = eq_of(last)
        start = state.get("initial_equity", curr)
        ret_pct = (curr / start - 1) * 100 if start else 0

        age = mins_since(last.get("bar_open_time"))
        stale = age is not None and age > STALE_MIN
        if stale:
            n_stale += 1
        else:
            n_active += 1

        total_eq += curr
        total_start += start

        marker = f"  [STALE {int(age)}m]" if stale else ""
        print(f"\n  {last.get('strategy', '?'):14s} {last.get('symbol', '?'):10s}{marker}")
        print(f"    Equity:   ${curr:,.2f}  (start ${start:,.2f}, {ret_pct:+.2f}%)")
        print(f"    Position: {last.get('current_pos_frac', 0):.2f}")
        print(f"    Trades:   {last.get('n_trades', 0)}  (W/L: {last.get('n_wins', 0)}/{last.get('n_losses', 0)})")
        print(f"    Risk:     {last.get('n_risk_rejections', 0)} rejections, "
              f"{last.get('n_risk_reductions', 0)} reductions")
        print(f"    Last bar: {last.get('bar_open_time', 'n/a')}")

    print(f"\n  {'─' * 68}")
    print(f"  PORTFOLIO")
    print(f"    Total equity:    ${total_eq:,.2f}")
    print(f"    Total start:     ${total_start:,.2f}")
    if total_start > 0:
        print(f"    Total return:    {(total_eq / total_start - 1) * 100:+.2f}%")
    print(f"    Active traders:  {n_active}")
    print(f"    Stale traders:   {n_stale}")

    ks = Path("research/KILL_SWITCH")
    print(f"    Kill switch:     {'PRESENT' if ks.exists() else 'not present'}")
    print()


if __name__ == "__main__":
    main()
    
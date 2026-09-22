"""Run the paper trader for EXP-004 (Donchian + RSI).

Press Ctrl+C to stop. State persists to disk; restart resumes from where
it left off.

Run:
    uv run python scripts/paper_trade_donchian_rsi.py
"""

from __future__ import annotations

from pathlib import Path

import yaml

from crypto_algo.paper.runner import run_forever


def main() -> None:
    with open("config/paper_trading.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    run_forever(cfg)


if __name__ == "__main__":
    main()
    
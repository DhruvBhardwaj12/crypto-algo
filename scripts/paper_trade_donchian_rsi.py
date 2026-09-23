"""Run the Donchian + RSI paper trader via the unified alt_runner.

Run:
    uv run python scripts/paper_trade_donchian_rsi.py
"""

from __future__ import annotations

import yaml

from crypto_algo.paper.alt_runner import run_forever


def main() -> None:
    with open("config/paper_trading.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    run_forever(cfg)


if __name__ == "__main__":
    main()
    
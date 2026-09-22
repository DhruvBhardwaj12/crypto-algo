"""Run the 5m RSI(2) paper trader (BTCUSDT) — LAB museum exhibit.

This strategy is KNOWN to lose money to cost drag. It is run purely to
observe live how a high-frequency signal bleeds equity.

Run:
    uv run python scripts/paper_trade_rsi2_5m.py
"""

from __future__ import annotations

import yaml

from crypto_algo.paper.alt_runner import run_forever


def main() -> None:
    with open("config/paper_trading_rsi2_5m.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    run_forever(cfg)


if __name__ == "__main__":
    main()
    
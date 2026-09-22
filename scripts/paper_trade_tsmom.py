"""Run the TSMOM paper trader (BTCUSDT, 4H, lookback=168).

Run:
    uv run python scripts/paper_trade_tsmom.py
"""

from __future__ import annotations

import yaml

from crypto_algo.paper.alt_runner import run_forever


def main() -> None:
    with open("config/paper_trading_tsmom.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    run_forever(cfg)


if __name__ == "__main__":
    main()
    
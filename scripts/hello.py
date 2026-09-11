"""First script — verifies the environment works end-to-end."""

from datetime import datetime, timezone

import numpy as np
import pandas as pd
from loguru import logger

from crypto_algo import __version__


def main() -> None:
    logger.info("crypto-algo version: {}", __version__)
    logger.info("Current UTC time: {}", datetime.now(timezone.utc).isoformat())

    # Tiny synthetic price series — proves pandas + numpy work
    rng = np.random.default_rng(seed=42)
    n = 100
    returns = rng.normal(loc=0.0005, scale=0.02, size=n)
    prices = 100.0 * np.exp(np.cumsum(returns))

    df = pd.DataFrame(
        {"close": prices},
        index=pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC"),
    )

    logger.info("Synthetic price series head:\n{}", df.head(3))
    logger.info("Mean price: {:.2f}", float(df["close"].mean()))
    logger.info("Std of returns: {:.4f}", float(np.std(returns)))

    logger.success("Environment OK. Ready for Milestone 3.")


if __name__ == "__main__":
    main()
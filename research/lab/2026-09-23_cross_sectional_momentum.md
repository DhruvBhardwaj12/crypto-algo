# LAB-20260923 — Cross-Sectional Momentum

**Hypothesis:** Rank a universe of 20 liquid crypto futures by trailing
7-day return. Long the top 3, short the bottom 3. Equal-weight, market-
neutral. Exploit the documented cross-sectional momentum effect.

**Mechanism:** Cross-sectional momentum has been documented in equities,
commodities, and currencies for decades. Crypto-specific drivers:
retail attention cycles, narrative flows, and limited arbitrage capital
between altcoin segments.

**Data:** 20 Binance USDT-M perpetual futures, 4H bars, 2020-01-17 to
2025-01-01. Symbols loaded: BTC, ETH, SOL, BNB, XRP, DOGE, ADA, AVAX,
LINK, DOT, MATIC, TRX, LTC, ATOM, NEAR, FIL, ARB, OP, INJ, SUI.

**Parameters (chosen before running, no tuning):**
- lookback = 42 bars (7 days)
- n_long = 3, n_short = 3
- rebalance_every = 6 bars (1 day)
- cost = 5 bps per side, 10 bps round trip

**Results:**
- Final equity: $181,879 (from $10,000)
- Total return: +1718.8%
- CAGR: 79.4%
- Sharpe: 1.67
- Sortino: 2.50
- Max drawdown: -32.1%
- Calmar: 2.47
- Annualized turnover: 230x
- Mean daily turnover (weight units): 0.105

**Verdict:** **PROMISING.** Best in-sample result of any strategy tested.
Requires full validation (parameter sweep, walk-forward, regime,
bootstrap) before consideration for paper trading.

**Known concerns (must be addressed in validation):**
1. **Survivorship bias.** Universe is 20 coins that exist today.
   Excluded coins that died (LUNA, FTT, etc.). If the strategy would
   have gone long a coin that later died, that loss is not in the data.
2. **High turnover.** 230x/year at 5 bps/side = ~11.5% annual cost drag.
   Real slippage on smaller-cap coins (ARB, OP, SUI, INJ) may exceed 5 bps.
3. **Small effective N.** Only 6 active positions out of 20. Sample of
   outcomes per rebalance is small.
4. **Dynamic universe.** Early bars (2020) had only 6 symbols. Later bars
   (2024) have 20. The strategy behavior changes over time.

**Next step:** Full validation framework. If it passes, paper trading.

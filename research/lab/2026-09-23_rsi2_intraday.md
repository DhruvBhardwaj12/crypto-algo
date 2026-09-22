# LAB-20260923 — RSI(2) Intraday Mean Reversion (5m)

**Hypothesis (one sentence):**
When 5-minute RSI(2) drops below 10, price is at a short-term panic
extreme and will revert upward over the following minutes.

**Mechanism (why should this edge exist?):**
Short-horizon mean reversion is documented in microstructure literature.
Liquidity providers demand compensation for absorbing sudden order-flow
imbalances, and stop-loss cascades overshoot fair value. Whether these
effects exceed retail transaction costs is the empirical question.

**Data used:**
BTCUSDT and ETHUSDT USDT-M perpetual futures, 5m bars,
2023-01-01 to 2025-01-01.

**Cost model:**
5 bps per side, 10 bps round trip.

**Parameters tested:**
  - rsi_period = 2
  - entry_threshold = 10
  - exit_threshold = 70
  (No parameter search — one fixed test.)

**In-sample result:**

BTC:
  - Sharpe: -13.80
  - CAGR:   -98.2%
  - Trades/year: 4,867
  - Trades/day:  13.3
  - Max DD: -99.97%
  - Profit factor: 0.697
  - Win rate: 56.0%

ETH:
  - Sharpe: -12.13
  - CAGR:   -98.8%
  - Trades/year: 4,857
  - Trades/day:  13.3
  - Max DD: -99.99%
  - Profit factor: 0.777
  - Win rate: 57.8%

**Verdict:** REJECTED — and this rejection is one of the most valuable
results we've produced.

**Next step:** None. Do not pursue intraday mean reversion on retail costs.

**Notes:**
The strategy's raw directional signal is real — win rate is 56-58%, well
above a coin flip. Backing out the cost component, the gross edge per
trade is roughly +1.7 bps (BTC). This is a genuine edge. It is also
about 6 times smaller than the round-trip cost of 10 bps.

The strategy would have made money without transaction costs. It lost
99.97% with them. This is the single clearest demonstration in this
project of the principle: a signal is only a strategy if its edge per
trade exceeds its cost per trade.

Every future experiment must state, before running, the expected edge
per trade and the cost per trade. If edge < cost, do not run it. This
lab has paid for itself by making that lesson visceral.

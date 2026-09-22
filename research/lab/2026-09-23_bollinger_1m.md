# LAB-20260923 — 1m Bollinger Band Mean Reversion

**Hypothesis:** When 1m close drops more than 2 std below its 20-bar SMA,
price reverts toward the mean within the next few minutes.

**Mechanism:** Same microstructure effects as RSI(2) at 5m — liquidity
providers demanding compensation for absorbing order-flow imbalances.

**Data:** BTCUSDT and ETHUSDT futures, 1m bars, 2024-10-01 to 2025-01-01.

**Cost model:** 5 bps per side, 10 bps round trip.

**Parameters:**
  - window = 20 (1m bars)
  - num_std = 2.0
  - Entry: close < lower band
  - Exit:  close > SMA

## Results

| Metric | BTC | ETH |
|---|---|---|
| Rows | 132,481 | 132,481 |
| Days covered | 92 | 92 |
| Trades/day | 24.8 | 25.1 |
| Win rate | 55.7% | 58.7% |
| Profit factor | 0.573 | 0.603 |
| Sharpe | −26.49 | −21.45 |
| Final equity | $1,312 | $1,995 |
| Total return | −86.9% | −89.1% |
| Buy & hold (same period) | +47.6% | +28.1% |

## Verdict: REJECTED — expected signature of cost drag on high-frequency.

The signal is directionally valid (55-59% win rate). The profit factor is
below 1 because losers are larger than winners — the classic mean reversion
tradeoff. But the ~1 bps gross edge per trade cannot cover the 10 bps
round-trip cost.

Cost drag per day: 25 trades × 10 bps = 250 bps = 2.5% of equity per day.
Extrapolated over 2 years: complete wipeout (matches LAB-002 pattern).

## Lesson reinforced

The frequency ladder:
- 4H: edge 20-50 bps vs cost 10 bps → survives
- 5m: edge 1-2 bps vs cost 10 bps → wiped
- 1m: edge 0.5-1.5 bps vs cost 10 bps → wiped

The threshold between "viable" and "dead" sits somewhere around the 15m-1H
range for retail crypto. Below that, no amount of signal cleverness
overcomes the arithmetic.

## Next steps

Close the high-frequency chapter. Focus on what works: 4H strategies.

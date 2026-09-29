# LAB-014 — Volatility Targeting Overlay on TSMOM

## Hypothesis
Vol clusters (proven by diagnostic: +0.23 lag-1 autocorrelation). Scale
position sizes inversely to realized vol to reduce drawdown and improve
Sharpe.

## Design (frozen, no tuning)
- Scale = min(1, trailing_median_vol / current_realized_vol)
- trailing_median_vol over 200 bars (5 weeks of 4H)
- current_realized_vol over 30 bars (5 days of 4H)
- Applied to TSMOM signals (BTC lb=168, ETH lb=84)
- No leverage: scale capped at 1.0

## Result

| | Sharpe base | Sharpe vol-target | DD base | DD vol-target |
|---|---|---|---|---|
| BTC historical | +1.512 | +1.374 | -43.4% | -39.0% |
| BTC recent | -0.267 | -0.411 | -36.9% | -33.8% |
| ETH historical | +1.382 | +1.137 | -40.5% | -42.9% |
| ETH recent | +0.614 | +0.502 | -49.9% | -43.4% |

**Sharpe declined in all four cases.** DD improved modestly (+3-6 pts).

## Verdict: REJECTED.

## Interpretation (the important part)

For crypto TSMOM, high-vol periods are the PROFITABLE periods. Cutting
exposure when vol spikes removes the strategy from its best trades.

This is consistent with the trend-following literature:
- Vol targeting HELPS mean-reversion (high vol = whipsaws)
- Vol targeting HURTS trend-following (high vol = big directional moves)

Our diagnostic showed vol is predictable, but predictable ≠ harmful.
We incorrectly assumed high vol was bad for returns. It is not.

## Key lesson
Predictability of a variable does not tell you its relationship to
strategy returns. We must test the DIRECTION of the correlation, not
just the magnitude.

## What this implies for future work
- Vol-targeting is NOT a universal overlay. Context matters.
- If we ever build a mean-reversion strategy that works, try vol
  targeting on THAT instead.
- Vol remains the most predictable feature we've measured. But using
  that predictability to improve TSMOM doesn't work.
  
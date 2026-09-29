# LAB-019 — Conditional Regime Switching

## Hypothesis
Rolling lag-1 autocorrelation of 4H returns can identify whether the
market is trending (positive autocorr) or ranging (negative autocorr).
Run TSMOM in trend regimes, RSI(2) reversion in range regimes, flat
otherwise.

## Rules (frozen, no tuning)
- Autocorr window = 500 bars (~83 days of 4H)
- Threshold = ±0.02
- TSMOM lookback = 42 bars
- RSI(2) entry < 10, exit > 50
- Shift by 1 bar for next-bar execution
- Cost: 5 bps per side

## Results

### Historical 2020-2025

| Symbol | TSMOM always | Reversion always | Regime switch | Switches/yr |
|---|---|---|---|---|
| BTC | +0.922 | -0.188 | +0.039 | 51 |
| ETH | +1.198 | +0.137 | +0.707 | 42 |

Per-regime diagnostics:
- BTC: TSMOM in "trend" +0.966, Reversion in "range" +0.861
- ETH: TSMOM in "trend" +1.933, Reversion in "range" +1.145

**Classifier works. Strategy underperforms base TSMOM.**

### Recent 2025-2026

| Symbol | TSMOM always | Reversion always | Regime switch | Switches/yr |
|---|---|---|---|---|
| BTC | +0.352 | -0.652 | -0.412 | 28 |
| ETH | +0.239 | -1.153 | -1.245 | 37 |

Per-regime diagnostics:
- BTC: TSMOM in "trend" +0.084, Reversion in "range" +2.971
- ETH: TSMOM in "trend" **-1.571**, Reversion in "range" **-1.883**

**Classifier fails on ETH. Both regimes show negative Sharpe.**

## Verdict: REJECTED.

The classifier carries real information — historical performance of each
strategy during its "correct" regime is clearly above unconditional
performance. But the strategy can't exploit this because:

1. Switching overhead (~30-50 flips/year at 10 bps round trip = 3-5%
   annual drag)
2. "Unclear" regime (33-42% of bars) means the strategy sits flat for
   over a third of the time, missing returns
3. Even when the classifier works, the combined strategy
   underperforms always-on TSMOM

In the recent regime, the classifier fails outright on ETH — both
regimes show negative Sharpe.

## The key diagnostic finding

For BTC in the recent window, "Reversion in range" Sharpe was +2.971.
That's a genuine signal. If we ran only that (long reversion during
range regime, flat otherwise), it would have produced positive returns.

But the strategy as constructed loses money because:
- It also runs TSMOM during "trend" periods (Sharpe +0.084)
- It sits flat during "unclear" periods (37.8% of bars)
- It switches between them ~28 times/year

The classifier's identification of "range" regimes is more valuable
than its identification of "trend" regimes. Asymmetric — worth
investigating separately.

## What this confirms

The autocorrelation regime signal is real but fragile. It fails
inconsistently across symbols in the recent window. Not a reliable
basis for a strategy.

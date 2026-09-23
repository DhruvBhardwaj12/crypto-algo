# LAB-20260923 — Recency Tests for All Validated Strategies

## Purpose

Test whether strategies that passed full historical validation on
2020-2025 still produce positive expectancy on the 21-month period
2025-01-01 to 2026-09-23, which was outside all prior validation windows.

This is the correct final test before any live trading decision. A
strategy that passes walk-forward on historical data but fails on fresh
data was likely capturing a regime-specific edge that has since ended.

## Methodology

- Frozen parameters from prior validation (no re-optimization)
- Same cost model (5, 10, 15 bps per side)
- 2025-01-01 to 2026-09-23 out-of-sample

## Results

### TSMOM (frozen: BTC lookback=168, ETH lookback=84)

| Symbol | bps/side | Total Ret | CAGR | Sharpe |
|---|---|---|---|---|
| BTC | 5  | -14.8% | -14.8% | -0.27 |
| BTC | 10 | -21.7% | -21.7% | -0.47 |
| BTC | 15 | -28.0% | -28.0% | -0.67 |
| ETH | 5  | +33.7% | +18.3% | +0.61 |
| ETH | 10 | +22.5% | +12.5% | +0.49 |
| ETH | 15 | +12.5% | +6.9%  | +0.37 |

**Verdict:** BTC failed, ETH passed. Mixed. Keep paper-trading ETH only.

### EXP-004 (Donchian + RSI, frozen: 55/20/52/48/50)

| Symbol | bps/side | Total Ret | Sharpe |
|---|---|---|---|
| BTC | 5  | -20.0% | -0.53 |
| BTC | 10 | -26.7% | -0.78 |
| BTC | 15 | -32.9% | -1.03 |
| ETH | 5  | -19.6% | -0.25 |
| ETH | 10 | -26.1% | -0.49 |
| ETH | 15 | -32.0% | -0.56 |

**Verdict:** FAILED both symbols. Stop paper trading.

### LAB-007 (Cross-sectional momentum)

| Cost | Sharpe |
|---|---|
| 5 bps  | +0.08 |
| 10 bps | -0.24 |
| 15 bps | -0.56 |

**Verdict:** FAILED. Rejected for live trading.

## Combined Assessment

**Zero of three validated strategies are proven to work in current
market conditions.** TSMOM on ETH is the only marginal survivor.

## Possible explanations (cannot distinguish from one sample)

1. **Regime change.** 2025-2026 may be a choppy, mean-reverting regime
   hostile to trend-following strategies.
2. **Edge decay from competition.** Crypto quant capital has grown.
   Documented edges compress.
3. **Overfitting.** The 2020-2025 validation may have captured noise
   specific to that period.

## Implications

- **Paper trading of EXP-004 should be stopped immediately.**
- **TSMOM on ETH may continue in paper** as the only signal of interest.
- **All future strategies must be tested on the recency window before
  being trusted.**
- **Adding this test as a permanent gate** in the validation framework.

## Next steps

1. Stop EXP-004 paper trader
2. Keep TSMOM ETH paper trading
3. Explore alternative strategy families (mean reversion, funding,
   on-chain flow, stat-arb)
4. Consider 3-month pause on new strategies to see if regime shifts back

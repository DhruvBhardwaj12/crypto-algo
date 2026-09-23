# LAB-20260923 — S&P 500 / BTC Cross-Asset Momentum

**Hypothesis:** During US market hours (13:00-20:00 UTC), BTC's next-hour
return is positively predicted by the S&P 500's recent return, via
cross-asset information flow through shared institutional participants.

**Mechanism:** Institutional investors trade both equities and crypto.
Risk-on / risk-off sentiment transmits from equities to crypto within
minutes. Documented in academic literature for traditional assets.

**Data:** SPY 1H bars (Yahoo Finance, 2023-10-24 to 2026-09-22, 5,072
bars). BTC USDT-M perpetual futures 1H bars (Binance, overlap with SPY
= 10,427 bars from 2023-10-24 to 2025-01-01).

**Cost model:** 5 bps per side, 10 bps round trip.

**Analysis 1 — Correlation (US hours only, n≈3,043):**
- SPY 1h return → next BTC 1h return: corr = +0.0102, t = +0.56
- SPY 3h return → next BTC 1h return: corr = −0.0044, t = −0.24
- Bonferroni threshold for 2 tests: |t| > 2.24
- **Both fail. Correlations indistinguishable from zero.**

**Analysis 2 — Simple strategy backtest:**
- Rule: long BTC when SPY 3h return > 0, during US hours. Flat otherwise.
- Final equity: $98.10 from $120 (total return −18.25%)
- Sharpe: −0.60
- Max drawdown: −34.11%
- Trades: 448 (~377/year)
- Win rate: 50.45%
- Profit factor: 1.02 (marginally above 1.0 — signal is very weakly
  positive gross of costs)
- Cost drag: 377 trades/year × 10 bps = 37.7% of equity per year

## Verdict: REJECTED.

The cross-asset momentum effect, at least in this form, does not exist
in the 2023-2025 data. The correlation is essentially zero (t = +0.56
and t = −0.24 — nowhere near significance). Even the weak positive
profit factor of 1.02 in the backtest is likely noise — it corresponds
to a gross edge of ~0.2 bps per trade, about 50x smaller than the cost
per round trip.

## Possible explanations

1. **The effect never existed mechanically.** The user's prior experience
   trading BTC by watching S&P was likely discretionary/intuitive and not
   a robust statistical pattern.
2. **The effect was arbitraged away.** If SPY → BTC transmission was
   tradable in 2023, sophisticated market participants may have
   eliminated it by 2024-2025.
3. **The effect exists at a different timeframe or in a specific
   regime** (e.g., VIX spikes, FOMC days), which our test did not
   isolate.

We cannot distinguish these from the data. What we can say is that
the simple US-hours version does not have a tradable edge at 1H
resolution.

## Lessons

1. **Discretionary observations do not automatically translate to
   mechanical edges.** The intuition "BTC responds to S&P" is not
   the same as "the effect is large enough and consistent enough to
   survive cost."
2. **Simple correlations are the right first test.** If two assets
   have correlation 0.01, no amount of strategy cleverness will make
   a profitable system. Always check the raw correlation before
   building a strategy.
3. **Bonferroni applies here too.** Two tests required |t| > 2.24. Neither
   passed. We avoided finding a "significant" effect that was just noise.

## Next steps

Close the cross-asset chapter. Focus energy on:
- Extending validated 4H strategies (EXP-004, TSMOM) to more symbols
- Building monthly review infrastructure for the running paper traders
- Reading and understanding the existing codebase

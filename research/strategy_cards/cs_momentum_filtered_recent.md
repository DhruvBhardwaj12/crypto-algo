## UPDATE 2026-09-23 — Extended test rejects the strategy

Extended the test window to 2024-01 to 2026-09 (2.73 years, includes
the ETF era from inception). Result:

| Metric | Strategy | BTC B&H |
|---|---|---|
| Total return | +40.1% | +104.5% |
| CAGR | +13.1% | +30.0% |
| Sharpe | +0.57 | +0.80 |
| Max DD | -42.9% | -53.4% |

**Rolling windows: 2/5 profitable.**
**Quarterly: 4/11 profitable.**
**Bootstrap 5th percentile: -0.663.**
**Worst quarter: 2025-Q1 (-31.82%).**

## Verdict: REJECTED.

The prior "marginal pass" on the 2025-01 to 2026-09 window was a
window-selection artifact. The 1.72-year test started after the
strategy's MA_200 warmup, which happened to exclude the worst quarter
(2025-Q1). When we extend back to 2024-01 and let the strategy be
fully active by early 2025, it loses 32% in a single quarter.

## Structural limitation

The 200-day MA regime filter is too slow. By the time BTC closes below
its 200-day MA, the drawdown is already 20-30% deep. The strategy
absorbs the drawdown on the way down before the filter turns off. This
is not fixable by parameter tuning — it's a design property of slow
regime filters.

## Lesson

**Backtests must include multiple starting points.** A strategy that
looks good on one window may look bad on another. Our walk-forward
methodology should have caught this, but it didn't because the recent
window was too short to expose the warmup artifact.

Going forward: any strategy on the recent regime must be tested on
BOTH the 1.7-year window AND the 2.7-year window. If results differ
materially, the shorter window is unreliable.

# Strategy Card — MA Crossover (Daily)

**Experiment ID:** EXP-002
**Date:** 2026-09-11
**Follows:** EXP-001 (hourly MA crossover, REJECTED)

## Hypothesis
Same as EXP-001. The 1h timeframe generated 60+ trades/year, consuming
~84% of equity to costs. We test whether a slower (daily) signal
reduces turnover enough that any underlying trend-following edge
becomes visible.

## Parameters
- fast_window = 20
- slow_window = 100
- timeframe = 1d
- period = 2020-01-01 to 2025-01-01 (5 years)
- Long-only, unlevered, 100% deployed when in signal
- (Parameters chosen BEFORE the test. No search performed.)

## Cost model
Buy 12.5 bps, sell 112.5 bps (round trip 125 bps). Over 5 years and 8
round trips, total cost drag ≈ 10%.

## Results

| Metric        | BTC      | ETH      | BTC B&H  | ETH B&H   |
|---------------|----------|----------|----------|-----------|
| Total return  | +849%    | +659%    | +1214%   | +2470%    |
| CAGR          | 59.9%    | 49.9%    | —        | —         |
| Sharpe        | 1.25     | 0.96     | —        | —         |
| Sortino       | 1.46     | 1.07     | —        | —         |
| Max drawdown  | -42.4%   | -76.1%   | —        | —         |
| Trades        | 8        | 8        | 0        | 0         |
| Win rate      | 50.0%    | 50.0%    | —        | —         |
| Profit factor | 3.11     | 1.78     | —        | —         |

Equity curve: `research/backtest_ma_daily.png`

## Robustness
Not yet tested. Single parameter set, single period, in-sample.

## Failure modes identified
- **Small sample size.** 8 trades over 5 years is statistically
  insufficient. Four winners is not evidence of edge.
- **Slow exit.** MA crossover exits only after the cross confirms, which
  means a large portion of a crash is absorbed before going flat. ETH's
  -76% DD demonstrates this.
- **Strong benchmark.** Buy & hold on crypto has been extraordinarily
  strong in this period. Any long-flat strategy that misses part of a
  bull market will underperform in absolute terms.

## Statistical concerns
- N=8 trades. No confidence interval on Sharpe is meaningful.
- Five-year period contains two bull markets and one major bear market.
  Performance in a *different* regime is unknown.
- No walk-forward, no out-of-sample, no parameter stability test.

## Verdict
**REJECTED.** Walk-forward analysis produced −24.46% compounded out-of-sample
return over 4 windows. Mean test Sharpe −0.09. Parameters chosen on prior data
do not transfer: 4 windows chose 4 different (fast, slow) pairs, and 2 of the
4 test periods lost money.

The full in-sample backtest (+849% BTC, +659% ETH) is not evidence of edge;
it reflects the strong 2020–2025 bull market lifting any long-biased strategy.
The parameter sweep (100% of the grid with Sharpe > 0.5) is a diagnostic of
smooth response surface, not a validity test.

## Why it failed
- **Parameter instability across regimes.** The optimal (fast, slow) pair
  changes every training window, indicating no persistent signal in the
  specific construction.
- **Long-biased beta, not alpha.** Calendar-year breakdown shows the strategy
  trails buy & hold in every up year and only outperforms in the 2022 bear
  market. This is a well-known return profile available more simply.
- **High-vol regime dependency.** Sharpe in high-vol regime = 0.38 vs
  mid-vol = 1.79. The strategy fails precisely when defensive value matters.

## What was learned
- The validation framework works. It rejected a strategy that looked good on
  the naive backtest.
- Trend-following on daily crypto may have some real content (100% of
  parameter grid positive over 5 years is unlikely to be pure luck), but
  single-asset, single-lookback, long-only MA is not the right construction.
- 8 trades over 5 years is not enough to conclude anything statistically.

## Statistical significance
- Bootstrap P(Sharpe > 0) = 99.2% (but bootstrap cannot detect regime change).
- Trade sign test p-value = 1.000 (sample size too small to conclude).
- Walk-forward: the only test that matters. Failed.


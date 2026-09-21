# Strategy Card — MA Crossover

**Experiment ID:** EXP-001
**Date:** 2026-09-11
**Author:** (your name)

## Hypothesis
When the fast moving average crosses above the slow moving average,
the asset is in an uptrend and should be held. When it crosses below,
exit to cash.

## Why should this edge exist?
Trend following is a documented phenomenon in many markets: prices
exhibit autocorrelation at medium horizons. In theory, a fast/slow MA
captures this. The counter-hypothesis is that crypto's noise-to-signal
ratio at hourly resolution, combined with high Indian transaction
costs, overwhelms any such edge.

## Market mechanism
Medium-horizon momentum in spot crypto.

## Required data
Hourly OHLCV, BTCUSDT and ETHUSDT, 2023-01-01 to 2024-01-01.

## Parameters tested
- fast_window = 20
- slow_window = 100
- timeframe = 1h
- (No parameter search performed — single fixed test.)

## Cost model
Buy: 12.5 bps. Sell: 112.5 bps. Round trip: 125 bps.

## Execution assumptions
Signals computed from close of bar i, executed at open of bar i+1.
Long-only, unlevered, 100% deploy when in signal.

## Results

| Metric           | BTC       | ETH       | BTC B&H  | ETH B&H  |
|------------------|-----------|-----------|----------|----------|
| Total return     | -21.07%   | -36.89%   | +156.96% | +92.24%  |
| Sharpe           | -0.50     | -1.11     | n/a      | n/a      |
| Max drawdown     | -54.78%   | -51.38%   | n/a      | n/a      |
| Trades           | 67        | 60        | 0        | 0        |
| Win rate         | 22.4%     | 25.0%     | n/a      | n/a      |
| Profit factor    | 0.91      | 0.69      | n/a      | n/a      |

Equity curve: `research/backtest_ma.png`

## Robustness
Not tested. Rejected before robustness testing because the base result
is unambiguously negative.

## Failure mode analysis
- **Cost drag:** 67 trades × 1.25% = 83.75% consumed by costs.
- **Whipsaw:** win rate ~22-25% indicates most signals are noise.
- **Wrong timeframe:** 20/100 MA on hourly is too fast; generates ~67
  crossings/yr against a very hostile cost regime.
- **Wrong benchmark period:** 2023 was a strong up year; buy & hold is
  a hard benchmark for any long-flat strategy that misses part of the move.

## Verdict
**REJECTED.** The strategy does not demonstrate a convincing edge under
realistic Indian cost assumptions. Not worth further refinement.

## Lessons carried forward
1. High-turnover strategies are structurally disadvantaged by 1% TDS.
2. Hourly crypto MA crossover is not a viable baseline for Indian retail.
3. Next investigations should focus on lower-frequency strategies
   (daily timeframe) and/or strategies with inherently fewer trades
   (funding rate, basis, cross-sectional ranking).
   
# Strategy Card — Multi-Timeframe Donchian + RSI

**Experiment ID:** EXP-004
**Date:** 2026-09-23
**Symbols:** BTCUSDT, ETHUSDT (Binance USDT-M perpetual futures)
**Timeframe:** 1H entry + 4H regime filter
**Period:** 2020-01-01 to 2025-01-01
**Cost model:** 5 bps per side (futures taker)

## Hypothesis
When the higher-timeframe (4H) RSI confirms bullish momentum, a 1H
Donchian breakout captures the start of sustained trends. The 4H filter
avoids trading against the dominant regime; the 1H breakout times entries.

## Mechanism
Trend persistence in crypto, on multi-day horizons, driven by momentum
and slow-moving capital flows.

## Rules
Entry (all true at 1H close):
  - 1H close > prior 55-bar 1H high
  - 1H RSI(14) > 50
  - 4H RSI(14) > 52 (most recent CLOSED 4H bar)

Exit (any true):
  - 1H close < prior 20-bar 1H low
  - 4H RSI(14) < 48

Position: long-only, target in {0, 1}, unlevered.

## Results (chosen params 55/20/52)

| Metric            | BTC       | ETH       | BTC B&H   | ETH B&H    |
|-------------------|-----------|-----------|-----------|------------|
| Total return      | +133%     | +227%     | +1216%    | +2511%     |
| CAGR              | 18.4%     | 26.7%     | —         | —          |
| Sharpe            | 0.67      | 0.76      | ~1.2      | ~1.1       |
| Max drawdown      | -50.9%    | -52.0%    | ~-75%     | ~-80%      |
| Calmar            | 0.36      | 0.51      | ~1.2      | ~1.1       |
| Trades (5 yr)     | 250       | 251       | 0         | 0          |
| Win rate          | 38.4%     | 38.2%     | —         | —          |
| Profit factor     | 1.26      | 1.20      | —         | —          |

## Validation results

**Parameter sweep:** 100% of 27 (dc × dc_exit × rsi4h) configs had positive
Sharpe on both symbols. Median Sharpe 0.96 (BTC) / 0.83 (ETH).
Best: dc=40, dc_exit=30, rsi4h=55.

**Walk-forward (5 folds, anchored):**
- BTC: compounded +194.5%, mean test Sharpe +0.97, 3/4 windows profitable
- ETH: compounded +94.5%, mean test Sharpe +0.50, 3/4 windows profitable
- BTC chose the same params (40,30,55) in all 4 windows → stable
- ETH params varied (75→40→55→40) → less stable but in a sensible region

**Regime breakdown (chosen params):**
- 2020: strong, but trailed buy & hold by 3x
- 2021: BTC negative (-24.1%) during a strong bull year — whipsaw cost
- 2022: both symbols protected capital (BTC -9.4% vs -64.2% benchmark)
- 2023-2024: positive but trailed buy & hold

**Bootstrap:** P(Sharpe > 0) = 93.1% (BTC), 94.7% (ETH). 5th percentile
Sharpe slightly negative on both.

**Trade sign test:** 250 trades, 38% win rate, p = 0.000. The win rate is
statistically BELOW 50%, but the profit factor > 1, meaning winners are
larger than losers on average. This is the expected trend-following shape.

## Failure modes identified
- Heavy dependence on 2020 bull run for aggregate return
- Weak Sharpe in high-volatility regimes (should be strongest there)
- Bootstrapped 5th percentile Sharpe near zero
- ~48 trades/year means cost drag is significant; real fills may be worse
- Underperforms buy & hold in every up year

## Verdict
**PASSED.** First strategy in the project to survive walk-forward validation.
Not investable yet. Requires paper trading before any live capital.

## Next steps
1. Paper trade for 4-8 weeks with the exact rules.
2. Compare expected vs realized fills.
3. Investigate whether a limit-order execution reduces cost drag.
4. Only then consider tiny live capital.

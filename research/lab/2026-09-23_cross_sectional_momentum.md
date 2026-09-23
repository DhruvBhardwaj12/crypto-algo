## Recent OOS test (2025-01-01 to 2026-09-23)

Frozen params (84/3/2/6 from 2020-2025 walk-forward). No re-optimization.

| bps/side | Total Ret | CAGR | Sharpe | Max DD |
|---|---|---|---|---|
| 5  | -1.9%   | -1.1%  | +0.08 | -33.2% |
| 10 | -14.7%  | -8.8%  | -0.24 | -38.6% |
| 15 | -25.8%  | -15.9% | -0.56 | -44.6% |
| 25 | -43.9%  | -28.5% | -1.21 | -55.9% |

**Verdict: REJECTED for live trading.** The strategy validated
historically but failed to make money in the 21 months following
the validation period.

## Interpretation

This is the cleanest demonstration in the project of edge decay.
The strategy passed:
- Parameter sweep (98% positive)
- Walk-forward (+1288% compounded, 4/4 windows)
- Regime breakdown (5/5 years positive)
- Bootstrap (5th percentile +0.999)
- Survivorship test (4 fixed universes, all positive)
- Cost sensitivity (survives 25 bps)

And still produced a near-zero result on fresh data. Historical
validation is not a guarantee of forward performance.

## Possible mechanisms for decay

1. **Competition.** Cross-sectional momentum is public knowledge.
   As capital flows into crypto quant, the edge compresses.
2. **Regime change.** Reduced dispersion among altcoins in 2025
   (BTC-dominant regime). Cross-sectional needs dispersion.
3. **Universe aging.** Our symbol list is now all highly correlated
   majors. Less structural dispersion than 2020.

We cannot distinguish these from a single period.

## Implication for other strategies

The two strategies currently paper-trading (EXP-004, TSMOM) must also
be tested on recent data before any live consideration. If they also
fail, we learn that the 2020-2025 period was favorable regime for
trend-following that has since ended.

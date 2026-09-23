# LAB-011 — Leveraged Trend Rider (Multi-Symbol)

## Hypothesis
Long-only trend following on 4H, leveraged, across 10 liquid majors.
Target: 3-5% monthly via leverage.

## Rules
- Entry: close > MA(50) AND RSI(14) > 45
- Exit: close < MA(50)
- Universe: BTC, ETH, SOL, BNB, XRP, DOGE, ADA, AVAX, LINK, DOT
- Leverage tested: 1x, 2x, 3x, 5x

## Results

### Historical (2020-2025)
| Lev | Return | Sharpe | Max DD | Avg Monthly | Liqs |
|---|---|---|---|---|---|
| 1x | +5766% | +13.04 | -58.7% | +9.08% | 0 |
| 2x | +11566% | +13.39 | -59.2% | +11.48% | 0 |
| 3x | -100% | -0.15 | -100% | -16.48% | 0 |
| 5x | -100% | -0.44 | -100% | -31.28% | 7 |

### Recent (2025-2026)
| Lev | Return | Sharpe | Max DD | Avg Monthly | Liqs |
|---|---|---|---|---|---|
| 1x | -33.6% | -0.68 | -59.2% | -1.88% | 0 |
| 2x | -67.7% | -0.64 | -86.2% | -3.22% | 0 |
| 3x | -58.0% | -0.44 | -94.4% | -4.54% | 0 |
| 5x | -97.2% | -0.44 | -99.5% | -6.81% | 22 |

## Verdict: REJECTED.

Negative edge in recent regime. Leverage multiplies the loss, not
the return. The historical Sharpe of 13 is a statistical artifact
from annualizing per-bar returns on 4H data with mostly-flat positions.

## Key lesson
Leverage is neutral. It multiplies edge. Negative edge × leverage =
bigger negative. Do not add leverage to a losing strategy.

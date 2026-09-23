# LAB-013 — Funding Carry (Final, Corrected Model)

## Purpose
Rebuild funding carry with proper basis-risk modeling. The first attempt
(EXP-003) had Sharpe 12, which was a modeling bug from ignoring basis.

## Data
- Spot: BTC/ETH 4H, 2020-2025 and 2025-2026
- Perp: BTC/ETH 4H, same windows
- Funding: BTC/ETH, same windows

## Method
Two-leg simulation: long spot + short perp. Track:
- Funding income (per 8h event)
- Basis P&L (change in spot-perp spread while in position)
- Entry + exit costs

## Results

### BTCUSDT
| Window | Funding | Basis | Costs | Net | Sharpe |
|---|---|---|---|---|---|
| Historical | +$657 | -$1,174 | $102 | **-8.4%** | -5.94 |
| Recent | +$169 | +$598 | $91 | **-9.5%** | -4.18 |

### ETHUSDT
| Window | Funding | Basis | Costs | Net | Sharpe |
|---|---|---|---|---|---|
| Historical | +$729 | -$1,674 | $124 | **-8.1%** | -2.86 |
| Recent | +$1,742 | -$3,282 | $141 | **-9.3%** | -0.99 |

## Verdict: REJECTED.

Funding income is real and positive in all windows. But basis P&L is
systematically negative and larger than the funding income in 3 of 4
windows.

## Mechanism (the real lesson)

Entering carry when funding is high means entering at maximum
spot-perp premium. The premium decays over the hold, and that decay
costs more than the funding collected. This is structural — not a
tuning issue.

The strategy is essentially: "buy the top of the basis, wait for
compression." That compression is exactly what funding income is
supposed to compensate — but at retail costs and with retail entry
timing, it doesn't.

## Why the literature disagrees

Documented funding carry Sharpe 1.5-2.0 comes from:
- Cross-exchange (not same-exchange) delta-neutral positions
- Continuous market-making, not directional entry
- Hundreds of simultaneous positions, not single-concentrated
- Institutional fee tiers

Our retail single-venue version is a different strategy with a
different (negative) expectancy.

## Recommendation

CLOSE the funding carry chapter permanently. Even a correct model shows
negative expectancy at retail costs and timing. Do not revisit unless
we reach institutional infrastructure.

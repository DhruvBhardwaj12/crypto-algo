# LAB-20260923 — Multi-Symbol 5m RSI(2)

**Purpose:** Test whether diversification across 8 symbols rescues a
high-frequency strategy that is individually broken.

**Data:** BTCUSDT, ETHUSDT, SOLUSDT, BNBUSDT, XRPUSDT, DOGEUSDT,
ADAUSDT, AVAXUSDT futures, 5m bars, 2024-07-01 to 2025-01-01.

**Cost:** 5 bps per side (10 bps round trip) — platform fee only, no GST,
no slippage.

**Construction:** Each symbol trades independently. Portfolio is equal-
weight average of normalized per-symbol equity curves.

## Results

| Symbol | Return | Sharpe | Win% | PF |
|---|---|---|---|---|
| BTCUSDT | -87.8% | -12.9 | 57.6% | 0.71 |
| ETHUSDT | -89.3% | -10.2 | 59.4% | 0.74 |
| SOLUSDT | -81.5% | -6.2 | 61.7% | 0.91 |
| BNBUSDT | -89.6% | -10.9 | 60.0% | 0.75 |
| XRPUSDT | -75.0% | -4.1 | 62.3% | 0.90 |
| DOGEUSDT | -82.2% | -4.8 | 61.6% | 0.90 |
| ADAUSDT | -82.3% | -5.3 | 61.6% | 0.84 |
| AVAXUSDT | -87.0% | -6.3 | 62.0% | 0.84 |

Portfolio: final $1,566 from $10,000 (-84.3%), Sharpe -8.24.

Average single-symbol Sharpe: -7.58.
Diversification benefit: **-0.65 (negative).**

Average pairwise correlation: **+0.59.**

## Verdict: REJECTED. Confirms the theoretical lesson.

Diversification does not improve Sharpe when the average pairwise
correlation is 0.59. With 8 symbols at ρ=0.59, the effective number of
independent bets is only ~1.67. You are not running 8 strategies; you
are running 1.67 copies of the same trade.

More importantly, diversification cannot change the SIGN of expectancy.
If each trade has negative expectancy, combining N such trades gives an
aggregate with negative expectancy. Variance goes down; mean stays
negative.

## Secondary finding

Altcoins (XRP, DOGE, ADA, SOL) show higher win rates (61-62%) and PF
closer to 1 (0.84-0.91) than BTC/ETH (57-59% wins, PF 0.71-0.74). Their
Sharpe is less negative as a result. This is a microstructure
observation — smaller-cap perps have bigger tick noise relative to the
reversion effect. It does not change the sign, only the slope.

## Key lesson

**Diversification is not a cure for a broken strategy.** It is a tool
for reducing variance of a strategy whose expectancy is already
positive. Applied to negative expectancy, it just distributes the pain
across more positions.

Corollary for future work: any attempt to build a "diversified crypto
portfolio" must recognize that BTC/ETH/SOL/majors have high pairwise
correlation (~0.6). Real diversification requires either asset classes
outside crypto or strategies with genuinely independent drivers.

## Chapter closed

The frequency ladder and diversification exploration is now complete:

| Experiment | Frequency | Outcome |
|---|---|---|
| EXP-004 Donchian+RSI | 4H | PASSED |
| LAB-001 TSMOM | 4H | PASSED |
| LAB-002 RSI(2) | 5m | REJECTED -99.97% |
| LAB-003B Bollinger | 1m | REJECTED -87% |
| LAB-003C Hour-of-day | — | REJECTED no signal |
| LAB-004 Multi-symbol 5m | 5m | REJECTED -84% (diversification failed) |

Pattern: only 4H strategies survive. Multi-symbol at high frequency does
not help.

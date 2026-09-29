# Lab Experiment Index

Complete log of every strategy tested and every diagnostic run.
Ordered chronologically. Each entry links to the full write-up.

Legend:
- **PASSED** — survived full validation (sweep + walk-forward + regime + bootstrap)
- **REJECTED** — failed validation, or failed recency test
- **DIAGNOSTIC** — measurement/exploration, no strategy
- **CLOSED** — abandoned due to modeling complexity

---

## Historical (2020-2025)

| ID | Name | Type | Verdict | Reason |
|---|---|---|---|---|
| EXP-001 | MA Crossover (1H) | Trend | REJECTED | 67 trades × 1.25% spot cost = 84% cost drag |
| EXP-002 | MA Crossover (1D) | Trend | REJECTED | Walk-forward -24.5% OOS |
| EXP-003 | Funding Carry v1 | Yield | REJECTED | Sharpe 12 = modeling bug (ignored basis) |
| **EXP-004** | **Donchian + RSI MTF** | **Trend** | **PASSED** | Walk-forward +194.5%/+94.5% |
| **LAB-001** | **TSMOM (4H)** | **Trend** | **PASSED** | Walk-forward +385.9%/+153.3% |
| LAB-002 | RSI(2) 5m | Mean-rev | REJECTED | -99.97% net, cost drag 480%/year |
| LAB-003B | Bollinger 1m | Mean-rev | REJECTED | -87% in 3 months |
| LAB-003C | Hour-of-Day | Calendar | REJECTED | Zero hours passed Bonferroni |
| LAB-004 | Multi-symbol RSI(2) 5m | Mean-rev | REJECTED | Diversification failed (ρ=0.59) |
| LAB-005 | SPY/BTC Cross-Asset | Cross-asset | REJECTED | Correlation ≈ 0 (t=0.56) |
| LAB-006 | Frequency Map | DIAGNOSTIC | — | Cliff at ~1-2H for retail |
| LAB-007 | Cross-Sectional Momentum | Cross-sect | PASSED* | *Historical Sharpe 1.67; failed recent |
| LAB-008 | Recency Tests (all) | DIAGNOSTIC | — | EXP-004 failed, TSMOM mixed, LAB-007 failed |
| LAB-009 | Combined Reversion Portfolio | Mean-rev | REJECTED | Walk-forward -5.1% |
| LAB-010 | CS Reversion alone | Mean-rev | REJECTED | Fragile; Sharpe +0.91 in-sample, dies at 15 bps |
| LAB-011 | Leveraged Trend Rider | Trend | REJECTED | All leverage levels negative recent |
| LAB-012 | Funding Carry v2 | Yield | CLOSED | Basis P&L bug; complexity too high |
| LAB-013 | Funding Carry final | Yield | REJECTED | Basis P&L swamps funding income |
| LAB-014 | Vol-Targeting Overlay | Overlay | REJECTED | Hurts TSMOM; high vol = profitable for trends |
| LAB-015 | Funding Z-Score Contrarian | Contrarian | REJECTED | Portfolio Sharpe ~0 recent |
| LAB-016 | Signal Scan | DIAGNOSTIC | — | taker_flow dead, funding_z weak, vol_ratio flipped |
| LAB-017 | Market Structure Diagnostic | DIAGNOSTIC | — | Return autocorr flipped; vol clustering strongest |

---

## Key Findings

### What Worked (historically, not recent)
- **EXP-004 (Donchian + RSI)** — Sharpe 0.67/0.76, walk-forward passed
- **LAB-001 (TSMOM)** — Sharpe 0.88/1.21, walk-forward passed
- **LAB-007 (Cross-sectional momentum)** — Sharpe 1.67, walk-forward +1288%

**All three fail in the recent (2025-2026) regime.**

### What Never Worked (any window)
- High-frequency (5m, 1m) — cost drag is mathematically fatal
- Funding carry (3 attempts) — basis risk or modeling complexity
- Vol targeting — hurts trend strategies
- Leveraged directional — negative edge × leverage = bigger losses
- Cross-asset (SPY/BTC) — no signal
- Calendar (hour-of-day) — no signal

### The Regime Shift (LAB-016, LAB-017)
- Return autocorrelation at 4H **flipped sign** (negative→positive)
- Vol autocorrelation **dropped 30%** but remains the strongest predictable signal
- Cross-sectional dispersion **dropped 30%** — alts move together more
- **Implication:** old edges (trend, mean-reversion) don't work in either direction

### The One Consistent Signal
- **Funding z-score** kept its direction (contrarian) across both windows, but the magnitude decayed to the point of zero net edge at portfolio level.

---

## What's Still Open

1. **vol_ratio as a standalone signal** — recent t = -8.69 at h=24 (strongest recent finding). Not yet tested as a strategy.
2. **Options-based strategies** — Deribit DVOL data available; volatility risk premium documented. Bigger infrastructure project.
3. **On-chain flow signals** — exchange netflow, whale accumulation, etc. Needs external data source.
4. **Regime-shift detection** — when does autocorrelation flip back? If we can time it, we can re-deploy old strategies.

---

## Method Rules Established

1. **Bonferroni correction** for multiple testing.
2. **Walk-forward is mandatory** before any strategy is trusted.
3. **Recency test on 2024-2026 data** before any strategy goes to paper trading.
4. **Cost model:** 5 bps/side futures, 125 bps round-trip spot (with TDS).
5. **Frozen parameters** — no tuning after seeing results.
6. **Every experiment logged**, including failures.
7. **No cherry-picking symbols or parameters** post-hoc.

---

## Paper Trading Status

| Strategy | Symbol | Status | Sharpe (historical) | Sharpe (recent) |
|---|---|---|---|---|
| TSMOM | ETHUSDT | Running | +1.21 | +0.61 |
| TSMOM | BTCUSDT | Paused | +0.88 | -0.27 |
| Donchian + RSI | BTCUSDT | Offline | +0.67 | -0.53 |
| RSI(2) 5m museum | BTCUSDT | Paused | -13.8 | — |

Only TSMOM-ETH shows positive recent-regime Sharpe.



# Market Structure Diagnostic — 2026-09-30

## Method
Statistical tests on 19 crypto futures, 4H bars.
Historical: 2020-2025 (10,963 bars). Recent: 2025-2026 (3,781 bars).

## Key findings

### 1. Return autocorrelation flipped sign at lag 1
Historical: BTC -0.034, ETH -0.039, SOL -0.039 (significant mean reversion)
Recent: BTC +0.021, ETH +0.043, SOL +0.046 (weak momentum)

**Implication:** The market no longer mean-reverts at 4H.
This explains why RSI(2) and pair-spread strategies failed recent.

### 2. Volatility autocorrelation dropped 30%
Historical: BTC +0.228, ETH +0.254, SOL +0.266
Recent: BTC +0.160, ETH +0.175, SOL +0.171

**Implication:** Vol is still the strongest predictable signal.
Vol-based sizing/filters should improve any strategy.

### 3. Cross-sectional dispersion dropped 30%
Historical mean: 122 bps. Recent mean: 85 bps.
Last 90 vs first 90: -8.4%.

**Implication:** Alts move together more. Cross-sectional strategies have
less edge.

### 4. BTC-ETH correlation stable at ~0.83
Both windows show ~0.83. The two majors remain one asset.

### 5. Variance ratio shows inconsistent structure
BTC shows weak reversion at long q (0.92), ETH shows weak momentum
(1.15), SOL shows reversion (0.92). No consistent pattern.

## What the data says

- The market is NOT a random walk. There is structure.
- But the direction of the short-term structure has flipped.
- The magnitude of every signal we measured has declined.
- **Volatility clustering is the most stable, most predictable feature.**

## Strategy implications

Directional strategies (trend, momentum, reversion) that worked in
2020-2025 have failed because the structure they exploited has flipped
or weakened. Testing more of them is unlikely to work until the regime
shifts back.

**The one signal with staying power is vol.** Strategies that exploit
vol predictability — position sizing, vol targeting, vol regime filters,
options selling — are the correct direction.

## Next steps

1. Add volatility targeting to existing strategies (low effort, potential
   Sharpe improvement without changing signal).
2. Investigate whether vol regime can be used as a switch between
   strategies.
3. Consider options infrastructure (bigger project) if we want to trade
   vol directly.
   
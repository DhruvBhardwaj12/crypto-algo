# LAB-20260923 — Hour-of-Day Seasonality

**Hypothesis:** Some hours of the day (UTC) have statistically
different average returns than others, and the pattern is stable
across time and symbols.

**Mechanism:** Funding times (00/08/16 UTC), US session open (13:30
UTC), Asian session low-liquidity (22-02 UTC), etc. might create
recurring directional pressure.

**Data:** BTCUSDT and ETHUSDT futures, 1H bars, 2020-2025.

**Method:** For each hour h ∈ [0, 23], compute mean 1H return, std,
t-statistic across all observations during hour h. Bonferroni threshold
|t| > 3.1 for 24 tests. Also split-half sign-agreement test and
year-by-year stability.

## Results

**BTC:**
- Zero hours pass Bonferroni.
- Split-half sign agreement: 13/24 (chance = 12).
- Year-by-year: sign flips randomly for every hour.
- "Best" hour (08): t = +1.81, fails single-test threshold even without
  correction.

**ETH:**
- Zero hours pass Bonferroni.
- Split-half sign agreement: 11/24 (worse than chance).
- Year-by-year: nothing stable.

## Verdict: REJECTED — no evidence of time-of-day seasonality.

The pattern expected from 48 hypothesis tests at α=0.05 is ~2 false
positives. We found zero that pass even the Bonferroni-corrected
threshold. The data is cleaner than pure chance; there is no effect.

## Lesson

1. Multiple testing matters. Testing 48 hypotheses requires stricter
   thresholds than a single test.
2. Sign-agreement over half-splits is a useful robustness check — a
   real edge keeps its sign, noise doesn't.
3. Most "patterns" in trading are like this. They look real in a
   single chart but vanish under multiple-testing correction.

## Rule for future lab experiments

Every exploration with more than ~3 hypotheses must report a
Bonferroni-corrected (or equivalent) significance threshold, not a
naive α=0.05. Log this in docs/decisions.md as a permanent rule.

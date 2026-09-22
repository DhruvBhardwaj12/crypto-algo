
## 2026-09-23 — Multiple-testing rule for lab experiments

**Decision:** Any lab experiment that tests more than 3 hypotheses
(e.g., 24 hourly means, 8 lookbacks, multiple symbols) must report
significance at a Bonferroni-corrected threshold, not at naive α=0.05.

**Reason:** LAB-003C tested 24 hourly means on 2 symbols = 48 hypotheses.
Naive α=0.05 would expect ~2 false positives. Without correction we'd
mistakenly promote noise to "signal."

**Rule:** if an effect doesn't survive Bonferroni (or equivalent), we
do not treat it as real. Log and move on.


## 2026-09-23 — Multiple-testing rule for lab experiments

**Decision:** Any lab experiment that tests more than 3 hypotheses
(e.g., 24 hourly means, 8 lookbacks, multiple symbols) must report
significance at a Bonferroni-corrected threshold, not at naive α=0.05.

**Reason:** LAB-003C tested 24 hourly means on 2 symbols = 48 hypotheses.
Naive α=0.05 would expect ~2 false positives. Without correction we'd
mistakenly promote noise to "signal."

**Rule:** if an effect doesn't survive Bonferroni (or equivalent), we
do not treat it as real. Log and move on.

## 2026-09-23 — File editing reliability

Recurring issue this session: multi-line pastes and edits sometimes fail
silently. The file appears correct in VS Code but the disk content is
stale or partial.

**Mitigation:** after any edit, run a content-specific verification from
PowerShell before using the file:

    Select-String -Path <file> -Pattern "<expected_unique_string>"

If the pattern doesn't appear, the edit didn't take. Re-do it.

**Preferred write path:** small edits only. For new large files, use
smaller incremental writes and verify each.

**Dashboard incident:** spent ~40 minutes debugging a dashboard that
turned out to be running a stale version of the file on disk. Repeated
edits "succeeded" in VS Code but never landed. Root cause unknown;
verification-by-content is the durable workaround.

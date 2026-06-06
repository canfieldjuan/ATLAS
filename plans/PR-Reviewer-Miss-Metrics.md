# PR-Reviewer-Miss-Metrics

## Why this slice exists

S1 added `REVIEW_MISSES.md` -- the reviewer-side flywheel ledger -- with the
rule "no escaped defect is fixed only once; it must become a gate." This slice
S4 of the review-workflow redesign (issue #1328) adds the metrics summarizer the
issue's section 2b named: it turns the ledger into reviewer-quality numbers,
especially the gap (b) signal "AI findings missed by a human reviewer," and can
enforce the every-miss-becomes-a-gate rule once the ledger has real entries.

## Scope (this PR)

Ownership lane: dev-workflow/review-contract
Slice phase: Workflow/process

1. Add `scripts/summarize_review_misses.py`: parse the Ledger table in
   `REVIEW_MISSES.md`, skip header/separator/seed rows, and report totals by
   who missed the defect (human / AI / CI), gated vs not-yet-gated, and AI
   findings a human reviewer missed. `--fail-on-ungated` makes it a gate.
2. Add `tests/test_summarize_review_misses.py`: fixtures for seed-only parsing,
   real-row parsing, the bucket/gated tallies, empty-cell detection, and the
   `--fail-on-ungated` exit-code contract.
3. Enroll the test in `.github/workflows/pre_push_audit.yml`.
4. Housekeeping: archive the merged S3 plan to `plans/archive/` and regenerate
   `plans/INDEX.md` (the section-1g teardown folded into this branch).

### Review Contract

- Acceptance criteria:
  - [ ] `scripts/summarize_review_misses.py` reports total / by-bucket / gated /
        AI-missed-by-human metrics and exits 0 on the current seed-only ledger.
  - [ ] `--fail-on-ungated` exits 1 when a logged miss names no gate, 0 when all
        are gated or none are logged.
  - [ ] Placeholder/seed rows are excluded from the metrics.
  - [ ] The S3 plan is archived and `plans/INDEX.md` regenerated.
- Affected surfaces: dev workflow tooling only. No product surface.
- Risk areas: mis-parsing a hand-edited ledger (over- or under-counting).
  Mitigated by the empty-cell and placeholder-row fixtures.
- Reviewer rules triggered: R2 (parser with a proven fail-on-ungated branch),
  R10 (maintainable). No path-trigger glob matches this slice's files.

### Files touched

- `.github/workflows/pre_push_audit.yml`
- `plans/INDEX.md`
- `plans/PR-Reviewer-Miss-Metrics.md`
- `plans/archive/PR-Reviewer-Path-Trigger-Audit.md`
- `scripts/summarize_review_misses.py`
- `tests/test_summarize_review_misses.py`

## Mechanism

`scripts/summarize_review_misses.py` slices the "Ledger" section of
`REVIEW_MISSES.md`, reads the markdown table rows, and drops header, separator,
and seed/placeholder rows (a row whose first two cells are empty or whose text
says "seed"/"first real entry"). Each remaining row maps to date / issue /
missed_by / root_cause / gate / owner. It classifies missed_by into human /
AI / CI buckets, counts a row as gated when its gate cell carries real content
(empty markers like a dash, "none", "none yet", or "tbd" count as ungated), and
flags an AI-finding-missed-by-human when a human-attributed row's text names a
bot/AI/Codex/Copilot finding. By default it prints the summary and exits 0;
`--fail-on-ungated` exits 1 if any logged miss has no gate, enforcing the ledger
rule. It is a report tool, so it is not wired into the blocking bundle.

## Intentional

- Not wired into `scripts/local_pr_review.sh`: it is a metrics report, not a
  per-PR gate, and the ledger is empty today, so a bundle line would be noise.
  The fixtures still run in CI so the parser is covered.
- `--fail-on-ungated` exists so the every-miss-becomes-a-gate rule can be turned
  into a scheduled/CI gate later without code changes, but is opt-in now.
- Common "no value" markers (none / none yet / tbd / n/a) count as ungated so a
  human-written placeholder gate cell is not mistaken for a real gate.

## Deferred

- Scheduling `--fail-on-ungated` in CI once the ledger has real entries.
- Trend/secondary metrics (turnaround, rework cycles) from issue section 2b that
  need data the ledger does not yet capture.

Parked hardening: none.

## Verification

- `tests/test_summarize_review_misses.py` -- fixtures pass (seed-only, real
  rows, buckets, gated/ungated, empty-cell, fail-on-ungated exit codes).
- `scripts/summarize_review_misses.py` on the live `REVIEW_MISSES.md` -- reports
  "no escaped defects logged yet" and exits 0.
- `scripts/check_ascii_python.sh` -- ASCII-clean.
- `scripts/local_pr_review.sh` -- full mechanical bundle green before push.

## Estimated diff size

| File | LOC |
|---|---:|
| `.github/workflows/pre_push_audit.yml` | 2 |
| `plans/INDEX.md` | 3 |
| `plans/PR-Reviewer-Miss-Metrics.md` | 92 |
| `plans/archive/PR-Reviewer-Path-Trigger-Audit.md` | 0 |
| `scripts/summarize_review_misses.py` | 174 |
| `tests/test_summarize_review_misses.py` | 97 |
| **Total** | **368** |

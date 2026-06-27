# PR-Deflection-Delta-Source-Window-Baseline

## Why this slice exists

#1316's delta-report design calls out baseline selection as product-critical:
monthly deltas should compare comparable ticket source windows, with an
immediate-prior fallback when source-date metadata is unavailable. The landed
delta path now has stable identity, a pure comparator, persistence, a paid read
surface, opt-in monthly automation, and overflow warnings, but
`select_previous_paid_report(...)` still chooses the newest paid report by
report `created_at` alone.

Root cause: the store boundary treats report generation time as the comparison
axis even though `deflection.v1.summary.source_date_start/end` is the product
truth for "what changed this period." This PR fixes the root at the shared
baseline-selection boundary so batch generation and paid read fallback prefer
the most recent paid report whose source window ends before the current source
window starts. If either side lacks complete source-date metadata, the existing
created-at previous-report fallback remains.

This branch also archives the just-merged #1852 plan as the AGENTS.md teardown
step.

Diff budget note: the code/test change is small, but the total lands just over
the soft cap because archiving #1852 refreshes the large generated
`plans/INDEX.md`.

## Scope (this PR)

Ownership lane: issue-1316/deflection-delta-source-window-baseline
Slice phase: Production hardening

1. Upgrade `select_previous_paid_report(...)` in the in-memory and Postgres
   stores to prefer a prior paid report by source-date window when the current
   and candidate models expose complete ISO source dates.
2. Preserve the current tenant/paid/current-created ordering fallback when
   source dates are absent or no earlier source window exists.
3. Add focused tests proving source-window preference, fallback behavior, and
   Postgres query shape.
4. Archive the merged #1852 plan and refresh the plans index.

### Review Contract

Acceptance criteria:
- A current report for a May source window prefers an April-window baseline
  over a more recently created same-window report.
- If the current report or every candidate lacks comparable source-date
  metadata, selection falls back to the existing newest paid report before the
  current report.
- Selection remains tenant-scoped, paid-only, and never chooses reports created
  after the current report.
- Batch generation and paid read fallback use the same store method; no second
  baseline-selection rule is introduced.

Affected surfaces:
- `extracted_content_pipeline.deflection_report_access`
- `tests.test_content_ops_deflection_delta_persistence`
- `plans/`

Risk areas:
- Invalid or missing source-date metadata causing false skips, SQL parser
  failures, or maturity-ratchet churn that pushes date validation into
  hand-rolled calendar logic.
- Accidentally weakening the paid/tenant/current-created boundary while adding
  source-window preference.
- Treating source-window preference as full calendar-month override support.

Reviewer rules triggered: R1, R2, R6, R8, R10, R14.

### Files touched

- `extracted_content_pipeline/deflection_report_access.py`
- `plans/INDEX.md`
- `plans/PR-Deflection-Delta-Source-Window-Baseline.md`
- `plans/archive/PR-Deflection-Delta-Paid-Overflow-Probes.md`
- `tests/maturity_sweep/baseline_deflection_lane.json`
- `tests/maturity_sweep/baseline_extracted_content_pipeline.json`
- `tests/test_content_ops_deflection_delta_persistence.py`

## Mechanism

- Add small helpers that extract strict calendar-valid `YYYY-MM-DD` source-date
  values from the sanitized persisted `deflection.v1` report model summary.
  Invalid or missing values return `None` through the standard library
  `date.fromisoformat(...)` parser inside a narrow `ValueError` guard.
- In-memory selection keeps the existing candidate set: same account, paid,
  not current, and `created_at < current.created_at`. Within that set it first
  looks for candidates whose `source_date_end` is before the current
  `source_date_start`, choosing the latest such source end. If none exist, it
  falls back to the existing newest-created candidate.
- Postgres mirrors the same logic using JSONB text extraction plus
  `to_date(...)` / `to_char(...)` round-trip validation. The `ORDER BY` ranks
  comparable source-window candidates first only after both dates are
  calendar-valid, then falls back to `created_at DESC`.
- `compute_and_save_previous_deflection_delta(...)` and
  `fetch_paid_deflection_delta(...)` already call
  `select_previous_paid_report(...)`, so the shared boundary upgrade covers
  generation and read fallback without duplicating rules.

## Intentional

- This is not full calendar-month override support. Explicit operator-selected
  baselines are already supported by the read API and generation override
  remains deferred.
- This keeps the existing `created_at < current.created_at` guard even when a
  later-created report has an earlier source window; a report generated after
  the current report is not a safe automatic baseline.
- The maturity baselines that cover this file are bumped for one justified
  parse-or-None guard. That is intentional: a narrow stdlib
  `date.fromisoformat(...)` guard is less brittle than duplicating calendar
  validation in Python and SQL.
- No customer-facing delta delivery, result page UI, entitlement logic,
  pagination, or live-cron enablement lands in this slice.

## Deferred

- Explicit baseline override for generation/backfills.
- Subscription-plan entitlement checks for which accounts should receive
  monthly deltas.
- Actual pagination across paid accounts and per-account reports before live
  production cron enablement.
- Monthly Report Delta delivery email, customer-facing copy, and result-page UI.

Parked hardening: none.

## Verification

- `python -m pytest tests/test_content_ops_deflection_delta_persistence.py -q`
  -- 23 passed.
- Python byte-compile check for `extracted_content_pipeline/deflection_report_access.py`
  -- passed.
- `git diff --check` -- passed.
- Deflection maturity sweep lane gate -- passed.
- General extracted-content maturity sweep ratchet gate -- passed.
- Plan sync and plan audits -- passed.
- Body-wired local PR review -- passed.

## Estimated diff size

| File | LOC |
|---|---:|
| `extracted_content_pipeline/deflection_report_access.py` | 168 |
| `plans/INDEX.md` | 141 |
| `plans/PR-Deflection-Delta-Source-Window-Baseline.md` | 149 |
| `plans/archive/PR-Deflection-Delta-Paid-Overflow-Probes.md` | 0 |
| `tests/maturity_sweep/baseline_deflection_lane.json` | 4 |
| `tests/maturity_sweep/baseline_extracted_content_pipeline.json` | 4 |
| `tests/test_content_ops_deflection_delta_persistence.py` | 178 |
| **Total** | **644** |

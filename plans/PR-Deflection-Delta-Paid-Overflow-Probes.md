# PR-Deflection-Delta-Paid-Overflow-Probes

## Why this slice exists

#1851 made the monthly deflection delta task warn when its paid account or
per-account report windows are exactly filled, but the reviewer called out the
remaining operator-quality gap: `count >= limit` is conservative and can fire
forever once a tenant base reaches the configured window, even when there is no
hidden work beyond the cap.

Root cause: the batch worker only sees the bounded rows it asks the store to
list. It can tell that a configured scan window filled, but it cannot prove
whether any paid account/report exists beyond that bounded window. This PR
fixes the root for observability by adding explicit count-beyond-limit probes
at the store boundary and carrying overflow signals into the task payload. It
does not add full pagination, enable live cron, or change customer-facing
delivery.

This branch also archives the just-merged #1851 plan as the AGENTS.md teardown
step.

Diff budget note: this slice sits at the soft cap because the store protocol
change needs in-memory, Postgres, batch, task, and focused test coverage, and
it folds the #1851 plan archive into the branch.

## Scope (this PR)

Ownership lane: issue-1316/deflection-delta-paid-overflow-probes
Slice phase: Production hardening

1. Add paid-account and paid-report count probes to the report artifact store
   contract and implementations.
2. Use those probes only when a configured account/report window fills, so the
   batch summary can distinguish exact-fill saturation from definite overflow.
3. Surface overflow fields in the autonomous task payload and return a distinct
   warning status when overflow is proven.
4. Archive the merged #1851 plan and refresh the plans index.

### Review Contract

Acceptance criteria:
- A batch that scans exactly the configured account/report limits but has no
  extra paid work keeps the new overflow fields false/empty while preserving
  the existing saturation fields.
- A batch with more paid accounts than `account_limit` sets account overflow.
- A batch with more paid reports for a scanned account than
  `reports_per_account` records that account in the report-overflow list.
- Existing disabled, DB-not-ready, no-reports, degraded, all-failed, saturated,
  and complete task behavior remains intact; overflow is a warning, not a
  failure or live-cron enablement.

Affected surfaces:
- `extracted_content_pipeline.deflection_report_access`
- `atlas_brain.autonomous.tasks.content_ops_deflection_delta_automation`
- `tests.test_content_ops_deflection_delta_persistence`
- `tests.test_deflection_delta_automation_task`
- `plans/`

Risk areas:
- Store protocol drift between in-memory and Postgres implementations.
- Adding count probes that run unnecessarily on underfilled windows.
- Confusing exact-fill saturation with definite hidden paid work.

Reviewer rules triggered: R1, R2, R6, R8, R10, R14.

### Files touched

- `atlas_brain/autonomous/tasks/content_ops_deflection_delta_automation.py`
- `extracted_content_pipeline/deflection_report_access.py`
- `plans/INDEX.md`
- `plans/PR-Deflection-Delta-Paid-Overflow-Probes.md`
- `plans/archive/PR-Deflection-Delta-Cap-Saturation-Signals.md`
- `tests/test_content_ops_deflection_delta_persistence.py`
- `tests/test_deflection_delta_automation_task.py`

## Mechanism

- Extend `DeflectionReportArtifactStore` with count probes for paid accounts
  and paid reports per account. In-memory derives counts from stored rows;
  Postgres uses `COUNT(DISTINCT account_id)` and account-scoped `COUNT(*)`.
- Extend `DeflectionDeltaBatchSummary` with defaulted overflow fields:
  `account_limit_overflow`, `reports_per_account_limit_overflow`, and
  `report_limit_overflow_accounts`.
- In `compute_and_save_recent_deflection_deltas`, keep the existing bounded
  listing behavior. When `account_limit_reached` is true, count total paid
  accounts and mark overflow only if the count is greater than the resolved
  limit. When a scanned account fills `reports_per_account`, count that
  account's paid reports and mark report overflow only when the count exceeds
  the resolved report limit.
- Include the overflow fields in the autonomous task payload. If no failures
  occurred and any overflow is proven, return a distinct `_skip_synthesis`
  message for scan-window overflow. Exact-fill saturation keeps the existing
  saturation warning.

## Intentional

- This is an overflow probe, not pagination. The worker still scans the same
  bounded account/report windows as before.
- Overflow is stricter than saturation: exactly N paid accounts/reports at an
  N-sized window is saturated but not overflowed.
- Count probes run only after a window fills, keeping underfilled monthly runs
  on the existing cheap path.
- No entitlement, email, customer-facing copy, or cron enablement changes land
  in this slice.

## Deferred

- Actual pagination across paid accounts and per-account reports before live
  production cron enablement.
- Subscription-plan entitlement checks for which accounts should receive
  monthly deltas.
- Monthly Report Delta delivery email and customer-facing copy.
- D0 stable identity foundation from #1316 remains the gate before
  customer-facing delta delivery.

Parked hardening: none.

## Verification

- `python -m pytest tests/test_content_ops_deflection_delta_persistence.py -q`
  -- 19 passed.
- `python -m pytest tests/test_deflection_delta_automation_task.py -q` -- 12
  passed.
- Python byte-compile check for the deflection report access module and
  deflection delta automation task -- passed.

## Estimated diff size

| File | LOC |
|---|---:|
| `atlas_brain/autonomous/tasks/content_ops_deflection_delta_automation.py` | 12 |
| `extracted_content_pipeline/deflection_report_access.py` | 93 |
| `plans/INDEX.md` | 3 |
| `plans/PR-Deflection-Delta-Paid-Overflow-Probes.md` | 138 |
| `plans/archive/PR-Deflection-Delta-Cap-Saturation-Signals.md` | 0 |
| `tests/test_content_ops_deflection_delta_persistence.py` | 112 |
| `tests/test_deflection_delta_automation_task.py` | 60 |
| **Total** | **418** |

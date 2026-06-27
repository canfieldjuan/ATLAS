# PR-Deflection-Delta-Cap-Saturation-Signals

## Why this slice exists

#1798 added the opt-in monthly deflection delta worker and #1850 made its paid
report scan prioritize paid activity, but both slices deferred the same live-cron
safety gap: if `account_limit` or `reports_per_account` is reached, the task
can complete without making the operator-visible result say that the bounded
window may have hidden paid work beyond the cap.

Root cause: the batch summary reports only scanned/saved/skipped/failed counts,
so the scheduler result cannot distinguish a complete scan from a scan that
filled one of its configured windows. This PR fixes the root for alerting by
carrying explicit cap-reached signals from the batch helper into the scheduled
task payload. It does not add pagination or enable live cron.

This branch also archives the just-merged #1850 plan as the AGENTS.md teardown
step.

## Scope (this PR)

Ownership lane: issue-1316/deflection-delta-cap-saturation-signals
Slice phase: Production hardening

1. Extend the delta batch summary with account/report window saturation fields.
2. Mark `account_limit` as reached when paid account discovery returns the
   configured account window.
3. Mark `reports_per_account` as reached per account when a paid report listing
   returns the configured report window.
4. Surface the new fields in the autonomous task result with a warning status.
5. Archive the merged #1850 plan and refresh the plans index.

### Review Contract

Acceptance criteria:
- A batch with fewer accounts/reports than the configured limits keeps the new
  saturation fields false/empty.
- A batch that reaches `account_limit` marks account saturation in the summary
  and task payload.
- A batch that reaches `reports_per_account` records the saturated account IDs
  and marks report-window saturation in the summary and task payload.
- Existing skip/degraded/all-failed behavior remains intact; saturation alone
  does not fail the task or enable live cron.

Affected surfaces:
- `extracted_content_pipeline.deflection_report_access`
- `atlas_brain.autonomous.tasks.content_ops_deflection_delta_automation`
- `tests.test_content_ops_deflection_delta_persistence`
- `tests.test_deflection_delta_automation_task`
- `plans/`

Risk areas:
- False-green monthly task results when bounded scans may be incomplete.
- Backward compatibility for existing `DeflectionDeltaBatchSummary` tests.
- Keeping alerting distinct from pagination/live cron enablement.

Reviewer rules triggered: R1, R2, R6, R8, R10, R14.

### Files touched

- `atlas_brain/autonomous/tasks/content_ops_deflection_delta_automation.py`
- `extracted_content_pipeline/deflection_report_access.py`
- `plans/INDEX.md`
- `plans/PR-Deflection-Delta-Cap-Saturation-Signals.md`
- `plans/archive/PR-Deflection-Delta-Paid-Report-Ordering.md`
- `tests/test_content_ops_deflection_delta_persistence.py`
- `tests/test_deflection_delta_automation_task.py`

## Mechanism

- Add defaulted fields to `DeflectionDeltaBatchSummary` for
  `account_limit_reached`, `reports_per_account_limit_reached`, and
  `report_limit_reached_accounts`.
- In `compute_and_save_recent_deflection_deltas`, normalize the configured caps
  through the same bounded-limit helper used by the store methods, then compare
  returned list lengths to those caps. Reaching the cap means "window filled,
  possible hidden work", not proof that hidden work exists.
- Include the saturation fields in the autonomous task payload. If there are no
  failures but at least one cap is reached, return a scheduler-visible
  `_skip_synthesis` message for scan-window saturation.

## Intentional

- This is a cap-reached warning, not pagination. The batch still scans the same
  number of accounts/reports as before.
- Reaching exactly the configured limit is treated as saturated. That can warn
  when there are exactly N paid accounts/reports, but the alert is intentionally
  conservative before live cron is enabled.
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
  -- 16 passed.
- `python -m pytest tests/test_deflection_delta_automation_task.py -q` -- 11
  passed.
- Python byte-compile check for the deflection report access module and
  deflection delta automation task -- passed.

## Estimated diff size

| File | LOC |
|---|---:|
| `atlas_brain/autonomous/tasks/content_ops_deflection_delta_automation.py` | 10 |
| `extracted_content_pipeline/deflection_report_access.py` | 23 |
| `plans/INDEX.md` | 3 |
| `plans/PR-Deflection-Delta-Cap-Saturation-Signals.md` | 124 |
| `plans/archive/PR-Deflection-Delta-Paid-Report-Ordering.md` | 0 |
| `tests/test_content_ops_deflection_delta_persistence.py` | 44 |
| `tests/test_deflection_delta_automation_task.py` | 48 |
| **Total** | **252** |

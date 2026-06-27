# PR-Deflection-Delta-Deferred-Summary

## Why this slice exists

#1863 made paid-state delivery conditions retryable by deferring the queue row
back to `pending`, but the run summary still reports those defers through the
same `failed` counter used for terminal send/content failures. The reviewer
called out the consequence: if every scanned row is temporarily unpaid, the task
can raise "delivery failed for all scanned deliveries" even though no delivery
is lost and the rows are waiting to retry.

Root cause: the row lifecycle now distinguishes terminal failure from retryable
deferral, but the in-memory summary contract did not gain the same distinction.
This slice fixes the root by carrying retryable defers as their own count from
the drain into the monthly automation status.

## Scope (this PR)

Ownership lane: deflection/report-deltas
Slice phase: Production hardening

1. Add a `deferred` count to the deflection delta delivery run summary.
2. Count retryable `source_report_not_paid` / `delta_no_longer_sendable` skips
   as deferred rather than failed in the drain.
3. Keep terminal failures (`missing_delivery_email`, `empty_delta_payload`,
   send exceptions) on the existing failed path.
4. Include `delivery_deferred` in the monthly automation payload and avoid the
   live all-failed raise when the run only deferred retryable rows.

### Review Contract

- Acceptance criteria:
  - Retryable paid-state skips increment `deferred`, leave rows `pending`, and
    do not increment `failed`.
  - Terminal delta delivery failures still increment `failed` and retain the
    existing degraded/total-failure behavior.
  - Monthly automation includes `delivery_deferred` in every payload shape.
  - A live run with only deferred rows reports delivery degraded instead of
    raising total delivery failure.
- Affected surfaces:
  - `atlas_brain/content_ops_deflection_delivery.py`
  - `atlas_brain/autonomous/tasks/content_ops_deflection_delta_automation.py`
  - focused delivery/task tests
- Risk areas:
  - Operator alert semantics for paid customer delivery.
  - Do not weaken incidents/logging or terminal failure visibility from #1863.
- Reviewer rules triggered: R1, R2, R6, R8, R10, R14.

### Files touched

- `atlas_brain/autonomous/tasks/content_ops_deflection_delta_automation.py`
- `atlas_brain/content_ops_deflection_delivery.py`
- `plans/INDEX.md`
- `plans/PR-Deflection-Delta-Deferred-Summary.md`
- `plans/archive/PR-Deflection-Delta-Live-Delivery.md`
- `tests/test_atlas_content_ops_deflection_delivery.py`
- `tests/test_deflection_delta_automation_task.py`

## Mechanism

Extend `DeflectionDeltaDeliveryRunSummary` with a defaulted `deferred` field so
existing tests/helpers that instantiate simple summaries remain source
compatible. In `send_pending_deflection_delta_deliveries`, `_defer_delta_delivery`
branches increment `deferred` and continue; `_fail_delta_delivery` branches keep
incrementing `failed`.

The monthly task forwards the new count as `delivery_deferred`. It still returns
a degraded delivery status whenever the drain reports terminal failures or
deferred work, but the "all scanned deliveries failed" raise only applies to
terminal failures. A run with only retryable defers is visible without paging as
a hard send outage.

## Intentional

- No database migration. The durable queue state remains `pending` for retryable
  defers; the new field is a run-summary/operator signal only.
- Keep the same incident emission from #1863 for deferred rows so operators can
  still see payment-lifecycle skips in logs/incidents.

## Deferred

- Rich delivery observability dashboards remain deferred from #1863; this slice
  only corrects the task summary semantics.

Parked hardening: none.

## Verification

- Passed:
  - python -m py_compile atlas_brain/content_ops_deflection_delivery.py atlas_brain/autonomous/tasks/content_ops_deflection_delta_automation.py tests/test_atlas_content_ops_deflection_delivery.py tests/test_deflection_delta_automation_task.py
  - pytest tests/test_atlas_content_ops_deflection_delivery.py tests/test_deflection_delta_automation_task.py (49 passed, 1 warning)
  - python scripts/sync_pr_plan.py plans/PR-Deflection-Delta-Deferred-Summary.md --check
- Pending before push:
  - bash scripts/push_pr.sh tmp/pr-1864-body.md --set-upstream origin claude/pr-deflection-delta-deferred-summary

## Estimated diff size

| File | LOC |
|---|---:|
| `atlas_brain/autonomous/tasks/content_ops_deflection_delta_automation.py` | 8 |
| `atlas_brain/content_ops_deflection_delivery.py` | 8 |
| `plans/INDEX.md` | 3 |
| `plans/PR-Deflection-Delta-Deferred-Summary.md` | 107 |
| `plans/archive/PR-Deflection-Delta-Live-Delivery.md` | 0 |
| `tests/test_atlas_content_ops_deflection_delivery.py` | 9 |
| `tests/test_deflection_delta_automation_task.py` | 64 |
| **Total** | **199** |

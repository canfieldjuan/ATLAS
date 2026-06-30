# PR-Deflection-Delta-Deferred-Status

## Why this slice exists

#1864 split retryable delta-delivery defers from terminal failures, but the
monthly task still reports a deferred-only run as "Deflection delta delivery
degraded." The reviewer marked this non-blocking but useful: a self-healing
pending-retry state should be visible without sounding like a hard send outage.

Root cause: the task status string still has only one non-success delivery
headline after the summary contract gained separate `failed` and `deferred`
counts. This slice fixes the root by giving deferred-only runs a distinct
operator status while preserving the terminal-failure and generation-failure
degraded/raise paths.

## Scope (this PR)

Ownership lane: deflection/report-deltas
Slice phase: Product polish

1. Return a distinct "Deflection delta delivery pending retries" status when a
   delivery drain only deferred retryable rows.
2. Keep "Deflection delta delivery degraded" for any terminal delivery failure.
3. Keep the live total-failure raise for terminal all-failed runs.
4. Add focused task tests proving deferred-only, mixed deferred+failed, and
   generation-failed plus deferred behavior.

### Review Contract

- Acceptance criteria:
  - Deferred-only runs return the pending-retries status and include
    `delivery_deferred`.
  - Runs with any terminal delivery failure still return degraded or raise,
    according to the existing #1864 terminal-failure guard.
  - Runs with generation failures and deferred-only delivery return automation
    degraded, not pending retries.
  - Successful and dry-run delivery paths remain unchanged.
- Affected surfaces:
  - `atlas_brain/autonomous/tasks/content_ops_deflection_delta_automation.py`
  - focused task tests
- Risk areas:
  - Operator status semantics for paid customer delivery.
  - Do not make terminal send/content failures look recoverable.
- Reviewer rules triggered: R1, R2, R6, R8, R10, R14.

### Files touched

- `atlas_brain/autonomous/tasks/content_ops_deflection_delta_automation.py`
- `plans/INDEX.md`
- `plans/PR-Deflection-Delta-Deferred-Status.md`
- `plans/archive/PR-Deflection-Delta-Deferred-Summary.md`
- `tests/test_deflection_delta_automation_task.py`

## Mechanism

After missing-config and generation-total-failure checks, the task evaluates
delivery summary outcomes in this order:

1. Terminal failures: preserve the current all-failed raise and degraded status.
2. Generation failures: preserve "Deflection delta automation degraded".
3. Deferred-only work: return "Deflection delta delivery pending retries".
4. Scan-window status / success: unchanged.

This leaves the new `delivery_deferred` payload field from #1864 as the count,
while the headline communicates whether the delivery state needs retry versus
operator intervention.

## Intentional

- This is status copy/control-flow only. It does not add dashboarding,
  alert routing, queue schema, or retry scheduling changes.
- Keep deferred rows visible in the task result instead of treating them as
  success; they still mean buyer delivery has not happened yet.

## Deferred

- Rich delivery observability dashboards remain deferred from #1863/#1864.

Parked hardening: none.

## Verification

- Passed:
  - python -m py_compile atlas_brain/autonomous/tasks/content_ops_deflection_delta_automation.py tests/test_deflection_delta_automation_task.py
  - pytest tests/test_deflection_delta_automation_task.py (20 passed, 1 warning)
- Pending before push:
  - python scripts/sync_pr_plan.py plans/PR-Deflection-Delta-Deferred-Status.md --check
  - bash scripts/push_pr.sh tmp/pr-1865-body.md --set-upstream origin claude/pr-deflection-delta-deferred-status

## Estimated diff size

| File | LOC |
|---|---:|
| `atlas_brain/autonomous/tasks/content_ops_deflection_delta_automation.py` | 9 |
| `plans/INDEX.md` | 3 |
| `plans/PR-Deflection-Delta-Deferred-Status.md` | 99 |
| `plans/archive/PR-Deflection-Delta-Deferred-Summary.md` | 0 |
| `tests/test_deflection_delta_automation_task.py` | 121 |
| **Total** | **232** |

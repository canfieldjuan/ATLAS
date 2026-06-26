# PR-Deflection-Delta-Monthly-Automation

## Why this slice exists

Issue #1316 has the paid report delta path split into small slices. #1795
persisted report-pair deltas and #1796 exposed the paid read surface, but the
system still has no scheduled entrypoint that keeps those deltas populated for
new paid reports. That leaves Report Deltas dependent on a one-off caller
remembering to run `compute_and_save_previous_deflection_delta`.

Root cause: the delta persistence primitive is scoped to one current report,
while the autonomous layer has no tenant-scoped batch worker that discovers
paid report accounts/reports and invokes the primitive on a schedule. This PR
fixes that root for generation only; customer-facing delivery/email copy stays
deferred.

Diff budget note: this is over the 400 LOC target because the slice crosses the
store contract, Postgres implementation, scheduler seed, task registration,
task handler, and focused tests that prove each boundary. Splitting the tests
from the wiring would leave an unprotected monthly task seed.

## Scope (this PR)

Ownership lane: issue-1316/deflection-delta-monthly-automation
Slice phase: Vertical slice

1. Add tenant/account discovery for paid deflection reports to the report
   artifact store contract and implementations.
2. Add a deterministic batch helper that walks paid reports per account and
   persists current-vs-previous deltas through the existing idempotent delta
   save path.
3. Register a disabled-by-default monthly autonomous builtin task that calls
   the batch helper when `ATLAS_DEFLECTION_DELTA_*` config enables it.
4. Prove registration, scheduler seed/config behavior, disabled/DB-not-ready
   task exits, tenant scoping, idempotent generation, and no-baseline skips.
5. Review follow-up: let real batch exceptions fail the scheduled task, rank
   paid account discovery by paid activity instead of report creation time, and
   enroll the automation task tests in PR CI.

### Review Contract

Acceptance criteria:
- The worker never scans reports cross-tenant; every report list and delta
  compute call is scoped by one discovered `account_id`.
- Existing paid/current report access rules remain the boundary: unpaid
  reports, missing baselines, and missing models do not create deltas.
- The autonomous task is disabled by default and uses typed settings, not raw
  environment reads.
- Re-running the worker is safe because it reuses the existing persisted-delta
  upsert path.

Affected surfaces:
- `extracted_content_pipeline.deflection_report_access`
- `atlas_brain.autonomous.scheduler`
- `atlas_brain.autonomous.tasks`
- `atlas_brain.config`

Risk areas:
- Store protocol drift between in-memory and Postgres implementations.
- Accidentally enabling a monthly task in seeded installs.
- Batch automation that silently ignores tenants or crosses tenant boundaries.

Reviewer rules triggered: R1, R6, R8, R10, R11, R12, R14.

### Files touched

- `atlas_brain/autonomous/scheduler.py`
- `atlas_brain/autonomous/tasks/__init__.py`
- `atlas_brain/autonomous/tasks/content_ops_deflection_delta_automation.py`
- `atlas_brain/config.py`
- `.github/workflows/atlas_content_ops_deflection_report_checks.yml`
- `extracted_content_pipeline/deflection_report_access.py`
- `plans/PR-Deflection-Delta-Monthly-Automation.md`
- `tests/test_content_ops_deflection_delta_persistence.py`
- `tests/test_deflection_delta_automation_task.py`

## Mechanism

- Extend the deflection report artifact store with a `list_paid_report_accounts`
  method. In-memory derives distinct paid-account IDs from stored records;
  Postgres reads distinct paid account IDs from `content_ops_deflection_reports`
  with a deterministic limit.
- Add `compute_and_save_recent_deflection_deltas`, which obtains paid accounts,
  lists recent paid reports for each account, and calls
  `compute_and_save_previous_deflection_delta` for each current report. It
  returns a summary with account/report/delta/skip/error counts.
- Add `atlas_brain.autonomous.tasks.content_ops_deflection_delta_automation`.
  The task exits before touching DB when disabled, exits safely when the DB pool
  is not initialized, and otherwise constructs the Postgres store and delegates
  to the batch helper. Real batch exceptions are logged and re-raised so the
  scheduler records a failed run instead of a false success.
- Seed and register `content_ops_deflection_delta_automation` as an opt-in
  monthly cron task, with cron/account/report limits resolved from typed
  `settings.deflection_delta`.
- Order paid account discovery by paid activity (`paid_at` with updated/created
  fallback) so newly unlocked older reports are eligible under the account
  limit.
- Enroll the automation task test in the content-ops deflection report workflow
  because it imports atlas autonomous/storage modules and should not run in the
  extracted pipeline's lighter dependency environment.

## Intentional

- No email delivery, customer-visible monthly digest copy, or subscription
  billing change in this slice. This only keeps persisted deltas ready for the
  later delivery/report-delta surfaces.
- The batch worker recomputes/upserts recent deltas instead of trying to detect
  a missing row first. That keeps the slice small and leans on the already
  idempotent persistence path.

## Deferred

- Monthly Report Delta delivery email and customer-facing copy.
- Subscription-plan entitlement checks for which accounts should receive
  monthly deltas.
- Live production cron enablement and operator runbook.
- The D0 stable identity foundation called out in #1316 remains the gate before
  any customer-facing delta delivery. This slice only keeps persisted pair
  deltas warm through the existing access boundary.

Parked hardening: none.

## Verification

- Focused delta pytest command for the persistence and automation task tests -- 17 passed, 1 torch CUDA warning.
- Scheduler trigger pytest command plus the automation task tests -- 16 passed, 1 torch CUDA warning.
- Python compile check for the changed delta access, automation task, scheduler, and config modules -- passed.
- Extracted content pipeline validation script -- passed.
- Extracted reasoning-import audit -- passed.
- Extracted standalone-debt audit -- passed.
- Extracted ASCII policy check -- passed.
- Pending before push: Plan sync check.
- Pending before push: Atlas push wrapper with the PR body file
  which runs the local PR review bundle once.
- Review follow-up verification is GitHub Actions only in this recovery
  session because the current Codex workspace has no local repo checkout,
  Python, `git`, or `gh`.

## Estimated diff size

| File | LOC |
|---|---:|
| `atlas_brain/autonomous/scheduler.py` | 17 |
| `atlas_brain/autonomous/tasks/__init__.py` | 1 |
| `atlas_brain/autonomous/tasks/content_ops_deflection_delta_automation.py` | 74 |
| `atlas_brain/config.py` | 14 |
| `.github/workflows/atlas_content_ops_deflection_report_checks.yml` | 9 |
| `extracted_content_pipeline/deflection_report_access.py` | 115 |
| `plans/PR-Deflection-Delta-Monthly-Automation.md` | 134 |
| `tests/test_content_ops_deflection_delta_persistence.py` | 85 |
| `tests/test_deflection_delta_automation_task.py` | 201 |
| **Total** | **650** |

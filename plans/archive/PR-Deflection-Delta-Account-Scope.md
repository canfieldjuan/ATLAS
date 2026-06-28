# PR-Deflection-Delta-Account-Scope

## Why this slice exists

#1866 landed the Report Delta go-live runbook and deliberately deferred manual
live sends for one buyer because the monthly task is not account-scoped. The
task can cap `account_limit=1`, but that still means "the newest paid account,"
not "this opted-in account," and the pending delta delivery drain also scans
globally.

Root cause: the generation batch and delta delivery drain are both global
unless bounded by count limits. A count limit is not an account ownership
boundary, and account-only scoping can still send more than the checked report
when one account has multiple paid reports or pending deltas. This slice fixes
the root by adding an explicit `target_account_id`/`account_id` plus optional
`current_request_id` metadata override that scopes generation and delivery
draining to the checked report.

The diff lands above the 400 LOC target because the review fix has four
load-bearing surfaces (generation, delivery drain/count, scheduler metadata,
and go-live runbook) plus the required #1866 plan teardown. The extra lines are
focused tests proving the full defect class: account-only scope, exact
current-report scope, blank metadata fail-closed behavior, and delivery
drain/count filtering. The real-Postgres live-claim proof is deferred until the
delivery CI gate has a Postgres service.

## Scope (this PR)

Ownership lane: deflection/report-deltas
Slice phase: Production hardening

1. Add optional account/current-report overrides to the recent delta batch
   helper.
2. Add optional account/current-report filters to the pending delta delivery
   drain/count.
3. Wire the scheduled task metadata through both generation and delivery.
4. Update the go-live runbook to use the account-scoped override for manual
   rehearsal/manual live-send guidance.
5. Archive the just-merged #1866 plan doc as teardown.

### Review Contract

- Acceptance criteria:
  - Without metadata, the monthly task keeps the existing global scheduled
    behavior.
  - With `target_account_id` or `account_id`, generation scans only that
    account; it must not fall back to newest paid accounts.
  - With `current_request_id`, generation scans only the checked current report.
  - With account/current-report scope, the delta delivery drain/count only sees
    rows for that account and current request.
  - A present-but-blank target metadata field fails closed instead of widening
    to a global run.
  - The task payload exposes the active `target_account_id` and
    `current_request_id` so an operator can prove the run was scoped.
  - Tests cover both the pure batch helper and scheduled task wiring.
- Affected surfaces:
  - Delta generation batch helper.
  - Delta delivery drain/count.
  - Monthly delta scheduled task.
  - Go-live runbook.
- Risk areas:
  - Money/customer email path activation.
  - Tenant/account scoping.
  - Scheduler metadata parsing.
Reviewer rules triggered: R1, R2, R3, R6, R8, R10, R14.

### Files touched

- `atlas_brain/autonomous/tasks/content_ops_deflection_delta_automation.py`
- `atlas_brain/content_ops_deflection_delivery.py`
- `docs/extraction/validation/content_ops_deflection_delta_go_live_runbook.md`
- `extracted_content_pipeline/deflection_report_access.py`
- `plans/INDEX.md`
- `plans/PR-Deflection-Delta-Account-Scope.md`
- `plans/archive/PR-Deflection-Delta-Go-Live-Runbook.md`
- `tests/test_atlas_content_ops_deflection_delivery.py`
- `tests/test_content_ops_deflection_delta_go_live_runbook.py`
- `tests/test_content_ops_deflection_delta_persistence.py`
- `tests/test_deflection_delta_automation_task.py`

## Mechanism

`compute_and_save_recent_deflection_deltas(...)` gains optional `account_id`
and `current_request_id` parameters. When `account_id` is present, it uses
exactly that one account instead of `list_paid_report_accounts(...)`;
account-limit overflow flags remain false because no discovery cap is in play.
When `current_request_id` is present, it loads exactly that paid report instead
of scanning the recent report window, so a manual go-live run cannot enqueue
multiple current-report deltas.

`send_pending_deflection_delta_deliveries(...)` and
`pending_deflection_delta_delivery_count(...)` gain optional `account_id` and
`current_request_id` parameters. The SQL keeps the existing global behavior
when they are null and adds account/current-report predicates when scoped.

`content_ops_deflection_delta_automation.run(...)` reads
`target_account_id` first, then `account_id`, plus `current_request_id`, from
task metadata. If present, it passes the target through generation, delivery
count, and delivery drain, includes it in the returned payload, and rejects
present-but-blank target values.

## Intentional

- No production default changes. The scheduled cron remains global when no
  target metadata is present.
- No delivery table migration. Existing rows already carry `account_id` and
  `current_request_id`; this slice filters by those columns.

## Deferred

- A customer-facing subscription entitlement model remains deferred. This slice
  only makes operator/manual activation safely scoped to a checked paid report.
- Real Postgres live-claim coverage for the delivery drain is deferred until
  the deflection delivery CI gate has a Postgres service. The string-level SQL
  guard remains in this PR, and the code path is otherwise covered by focused
  unit tests.

Parked hardening: none.

## Verification

- Passed locally:
  - python -m py_compile extracted_content_pipeline/deflection_report_access.py atlas_brain/content_ops_deflection_delivery.py atlas_brain/autonomous/tasks/content_ops_deflection_delta_automation.py tests/test_content_ops_deflection_delta_persistence.py tests/test_atlas_content_ops_deflection_delivery.py tests/test_deflection_delta_automation_task.py tests/test_content_ops_deflection_delta_go_live_runbook.py
  - pytest tests/test_content_ops_deflection_delta_persistence.py tests/test_atlas_content_ops_deflection_delivery.py tests/test_deflection_delta_automation_task.py tests/test_content_ops_deflection_delta_go_live_runbook.py
    - 87 passed, 1 warning.
- Passed locally:
  - python scripts/sync_pr_plan.py plans/PR-Deflection-Delta-Account-Scope.md --check

## Estimated diff size

| File | LOC |
|---|---:|
| `atlas_brain/autonomous/tasks/content_ops_deflection_delta_automation.py` | 56 |
| `atlas_brain/content_ops_deflection_delivery.py` | 41 |
| `docs/extraction/validation/content_ops_deflection_delta_go_live_runbook.md` | 47 |
| `extracted_content_pipeline/deflection_report_access.py` | 66 |
| `plans/INDEX.md` | 1 |
| `plans/PR-Deflection-Delta-Account-Scope.md` | 144 |
| `plans/archive/PR-Deflection-Delta-Go-Live-Runbook.md` | 0 |
| `tests/test_atlas_content_ops_deflection_delivery.py` | 65 |
| `tests/test_content_ops_deflection_delta_go_live_runbook.py` | 18 |
| `tests/test_content_ops_deflection_delta_persistence.py` | 129 |
| `tests/test_deflection_delta_automation_task.py` | 167 |
| **Total** | **734** |

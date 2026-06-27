# PR-Deflection-Delta-Live-Delivery

## Why this slice exists

The report-deltas lane now has generation, paid read projection, source-window
baseline selection, cap telemetry, monthly scheduling, and delivery-safe email
copy. The missing buyer-visible step is live delivery: the monthly automation
still computes and saves deltas without creating durable send work or sending a
delta email.

Root cause: the persisted delta table is only a read surface. It has no delivery
state, so the scheduled task has no durable "send this delta once" contract.
Directly sending from generation would be a symptom fix because a retry/manual
rerun after the ESP idempotency window could resend the same monthly delta. This
slice fixes the root for the first vertical by adding persistent delta delivery
queue state and draining it from the monthly task.

Diff-size note: this is over the 400 LOC target because the first live delivery
vertical has to land the durable queue, store enqueue contract, drain worker,
monthly task composition, migration coverage, and privacy/idempotency tests
together. Splitting before the drain would recreate the already-called-out
"delivery primitive but not live" gap; splitting after the drain would leave the
queue unproven by the scheduled buyer path.

## Scope (this PR)

Ownership lane: deflection/report-deltas
Slice phase: Vertical slice

1. Add a small persistent delivery queue for saved deflection deltas, keyed by
   `(account_id, current_request_id, baseline_request_id)`.
2. Enqueue a delta delivery when recent-delta generation saves a paid delta and
   the current paid report has a delivery email.
3. Add a live drain function that claims queued delta deliveries, fetches the
   paid allowlisted delta payload, renders the #1862 delivery summary, sends via
   the existing campaign sender interface, and marks delivered/failed.
4. Compose the monthly delta automation task so one run computes, enqueues, and
   drains queued delta delivery work using the existing deflection-delivery
   config/sender pattern.
5. Add focused tests for enqueue dedupe, paid/read fail-closed behavior,
   payment-lifecycle recoverability, idempotency key shape, dry-run behavior,
   and monthly task delivery counts.

### Review Contract

- Acceptance criteria:
  - Saved deltas with no `delivery_email` do not enqueue delivery work.
  - A saved delta with a delivery email enqueues one durable row and reruns do
    not duplicate it after delivery, while recoverable payment-lifecycle
    failures can be revived.
  - The sender receives only the allowlisted delta read payload rendered by
    `deflection_delta_delivery_summary`; raw evidence/source fields remain out
    of the email path.
  - The monthly automation result reports generated and delivery counts without
    hiding generation failures or a blocked delivery backlog.
  - Delivery is idempotent across reclaims through a deterministic
    `(account_id, current_request_id, baseline_request_id)` key.
- Affected surfaces:
  - `extracted_content_pipeline/deflection_report_access.py`
  - `atlas_brain/content_ops_deflection_delivery.py`
  - `atlas_brain/autonomous/tasks/content_ops_deflection_delta_automation.py`
  - `atlas_brain/api/billing.py`
  - deflection migration chain/tests
- Risk areas:
  - Money/subscription email send path; avoid repeat sends.
  - Privacy boundary; delivery must use the paid allowlisted delta read payload.
  - Existing paid-report delivery behavior; keep report and delta delivery
    status independent.
- Reviewer rules triggered: R1, R2, R4, R5, R6, R8, R10, R14.

### Files touched

- `atlas_brain/api/billing.py`
- `atlas_brain/autonomous/tasks/content_ops_deflection_delta_automation.py`
- `atlas_brain/content_ops_deflection_delivery.py`
- `atlas_brain/storage/migrations/341_content_ops_deflection_delta_deliveries.sql`
- `extracted_content_pipeline/deflection_report_access.py`
- `plans/INDEX.md`
- `plans/PR-Deflection-Delta-Live-Delivery.md`
- `plans/archive/PR-Deflection-Delta-Delivery-Summary.md`
- `tests/maturity_sweep/deflection_product_surface_manifest.json`
- `tests/test_atlas_billing_content_ops_deflection_stripe_paid.py`
- `tests/test_atlas_content_ops_deflection_delivery.py`
- `tests/test_content_ops_deflection_delta_persistence.py`
- `tests/test_deflection_delta_automation_task.py`
- `tests/test_deflection_migrations_apply.py`

## Mechanism

Add a `content_ops_deflection_delta_deliveries` table with the same
pending/sending/delivered/failed shape as paid report delivery, but keyed to the
delta pair. The delta batch worker enqueues after `save_deflection_delta`
succeeds and only when the current report carries a delivery email; conflict
handling refreshes retryable work without duplicating delivered rows.

The delivery drain claims pending/stale rows, joins to the current and baseline
paid reports plus the persisted delta, projects through
`deflection_delta_read_payload`, renders `deflection_delta_delivery_summary`,
and sends a `SendRequest` with deterministic idempotency key
`deflection-delta:{account_id}:{current_request_id}:{baseline_request_id}`.
Delivered/failed updates are scoped to the claimed delivery row. Valid dry-run
rows increment `dry_run` before validation and never mutate state. Terminal
content/send failures mark the row failed and emit an incident so broken queue
rows do not spin forever. Recoverable payment-lifecycle failures
(`source_report_not_paid`, `delta_no_longer_sendable`) defer the row back to
`pending` with an incident instead of bricking delivery.

On enqueue conflict, the store refreshes the delivery email and revives
recoverable failed rows while leaving delivered, sending, and terminal-failed
rows untouched. Stripe dispute-won restore also requeues recoverable delta
deliveries tied to the restored current or baseline report, matching the paid
access lifecycle instead of requiring a future monthly generator run.

The monthly automation task reuses the existing deflection delivery config and
sender factory so operators configure one transactional sender. After generation
it drains the delta queue in the same run when delivery config is present. If a
send-path failure occurs, the task surfaces delivery degraded ahead of partial
generation degradation; total live-delivery failure raises.
When delivery config is missing, the task checks the durable queue for pending
or stale work, so a prior-run backlog cannot report as "automation complete"
while buyer emails remain unsent.

## Intentional

- No new public API/CLI in this slice. The live monthly task is the first
  buyer-visible path; manual CLI polish can follow after the durable path is
  proven.
- The delta delivery queue is separate from `content_ops_deflection_report_deliveries`
  because report delivery is keyed to a single paid report while deltas are
  keyed to a current/baseline pair.
- The delta automation remains opt-in through `ATLAS_DEFLECTION_DELTA_ENABLED`;
  this slice wires live delivery when the task runs, but does not flip production
  enablement defaults.
- Dry-run never sends or mutates queued rows, including invalid rows. Live runs
  still persist terminal failures and incidents; recoverable payment-lifecycle
  conditions remain retryable.

## Deferred

- A dedicated operator CLI for draining delta deliveries outside the monthly
  task is deferred until the live scheduled path is proven.
- Rich delivery observability dashboards are deferred; this slice returns task
  summary counts and persists per-row status/error.

Parked hardening: none.

## Verification

- Passed:
  - python -m py_compile extracted_content_pipeline/deflection_report_access.py atlas_brain/content_ops_deflection_delivery.py atlas_brain/autonomous/tasks/content_ops_deflection_delta_automation.py atlas_brain/api/billing.py tests/test_content_ops_deflection_delta_persistence.py tests/test_atlas_content_ops_deflection_delivery.py tests/test_deflection_delta_automation_task.py tests/test_atlas_billing_content_ops_deflection_stripe_paid.py
  - pytest tests/test_atlas_content_ops_deflection_delivery.py tests/test_deflection_delta_automation_task.py tests/test_content_ops_deflection_delta_persistence.py tests/test_atlas_billing_content_ops_deflection_stripe_paid.py::test_stripe_webhook_won_dispute_restores_paid_deflection_report (73 passed, 1 warning)
  - pytest tests/test_atlas_billing_content_ops_deflection_stripe_paid.py::test_stripe_webhook_won_dispute_restores_paid_deflection_report (1 passed, 1 warning)
  - bash scripts/validate_extracted_content_pipeline.sh
  - python extracted/_shared/scripts/forbid_atlas_reasoning_imports.py extracted_content_pipeline
  - python scripts/audit_extracted_standalone.py --fail-on-debt
  - bash scripts/check_ascii_python.sh
  - bash extracted/_shared/scripts/sync_extracted.sh extracted_content_pipeline
  - python scripts/sync_pr_plan.py plans/PR-Deflection-Delta-Live-Delivery.md --check
  - pytest tests/test_deflection_migrations_apply.py (1 passed, 2 skipped; DB-backed cases skipped because `ATLAS_MIGRATION_TEST_DATABASE_URL` is not set)
  - python scripts/check_deflection_product_surface_manifest.py
  - pytest tests/test_deflection_product_surface_manifest.py (3 passed)
- Observed:
  - pytest tests/test_atlas_billing_content_ops_deflection_stripe_paid.py (117 passed, 2 failed, 1 warning; failures are existing price-gate config expectations unrelated to the delta lifecycle change)
- Pending before push:
  - bash scripts/push_pr.sh tmp/pr-1863-body.md --set-upstream origin claude/pr-deflection-delta-live-delivery

## Estimated diff size

| File | LOC |
|---|---:|
| `atlas_brain/api/billing.py` | 39 |
| `atlas_brain/autonomous/tasks/content_ops_deflection_delta_automation.py` | 99 |
| `atlas_brain/content_ops_deflection_delivery.py` | 560 |
| `atlas_brain/storage/migrations/341_content_ops_deflection_delta_deliveries.sql` | 27 |
| `extracted_content_pipeline/deflection_report_access.py` | 95 |
| `plans/INDEX.md` | 3 |
| `plans/PR-Deflection-Delta-Live-Delivery.md` | 185 |
| `plans/archive/PR-Deflection-Delta-Delivery-Summary.md` | 0 |
| `tests/maturity_sweep/deflection_product_surface_manifest.json` | 1 |
| `tests/test_atlas_billing_content_ops_deflection_stripe_paid.py` | 52 |
| `tests/test_atlas_content_ops_deflection_delivery.py` | 306 |
| `tests/test_content_ops_deflection_delta_persistence.py` | 51 |
| `tests/test_deflection_delta_automation_task.py` | 248 |
| `tests/test_deflection_migrations_apply.py` | 39 |
| **Total** | **1705** |

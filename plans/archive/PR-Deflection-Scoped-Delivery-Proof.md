# PR-Deflection-Scoped-Delivery-Proof

## Why this slice exists

#1921 PR 4 is the deployed launch proof: Snapshot email/PDF, real Checkout
unlock, paid report email/PDF, and the exact emailed hosted URL. The current
runbook still has a brittle paid-delivery step because
`scripts/send_content_ops_deflection_report_deliveries.py` can only drain the
global paid-report delivery queue. To avoid emailing the wrong buyer, the
runbook must prove the target request is the only pending/sending row before
live send.

Root cause: the manual launch-proof drain has no account/request scope even
though the delivery row identity is `(account_id, request_id)`. This change
fixes the root for the operator proof path by adding paid-report delivery
scope filters to the shared worker query and exposing them in the manual CLI.
The scheduler remains queue-wide by default.

## Scope (this PR)

Ownership lane: content-ops/deflection-launch-readiness
Slice phase: Production hardening

1. Add optional paid-report delivery scope filters for `account_id` and
   `request_id` to `send_pending_deflection_report_deliveries(...)`.
2. Expose those filters through the manual paid-delivery CLI as
   `--account-id` and `--request-id`.
3. Update the launch preflight runbook to use the scoped dry-run/send commands
   and keep the global queue-empty check only as a scheduler safety gate.
4. Add focused tests proving both dry-run and claim queries carry the scope
   filters, the CLI passes filters into the worker, and the unscoped scheduler
   path remains unchanged.

### Review Contract

- Acceptance criteria:
  - A scoped dry-run query filters by account/request without claiming rows.
  - A scoped live-send claim query filters by account/request before
    `FOR UPDATE SKIP LOCKED`.
  - The manual CLI passes `--account-id`/`--request-id` to the worker and keeps
    existing behavior when the flags are omitted.
  - The autonomous delivery task remains unscoped and queue-wide.
  - The runbook no longer requires target-row global uniqueness before the
    manual proof send; it still requires zero claimable rows before probing the
    deployed scheduler.
- Affected surfaces:
  - Paid report delivery worker selection.
  - Manual launch-proof delivery CLI.
  - Launch preflight runbook.
- Risk areas:
  - Paid buyer email delivery.
  - Launch proof accidentally sending a non-target report.
  - Scheduler delivery behavior changing unexpectedly.
- Triggered reviewer rules:
  - R1 Requirements match.
  - R2 Test evidence.
  - R3 Security/auth/money path.
  - R8 Persistence/data lifecycle.
  - R14 Codebase verification.

### Files touched

- `atlas_brain/content_ops_deflection_delivery.py`
- `docs/extraction/validation/content_ops_deflection_launch_preflight_runbook.md`
- `plans/PR-Deflection-Scoped-Delivery-Proof.md`
- `scripts/send_content_ops_deflection_report_deliveries.py`
- `tests/test_atlas_content_ops_deflection_delivery.py`
- `tests/test_content_ops_deflection_launch_preflight_runbook.py`
- `tests/test_send_content_ops_deflection_report_deliveries.py`

## Mechanism

Extend `send_pending_deflection_report_deliveries(...)` with optional
`account_id` and `request_id` keyword arguments. Normalize blank values to
`None`, pass them into both pending-selection SQL statements, and add
`($2::text IS NULL OR d.account_id = $2)` plus
`($3::text IS NULL OR d.request_id = $3)` predicates before ordering/claiming.
Existing callers omit the filters and receive the same queue-wide behavior.

The CLI adds `--account-id` and `--request-id`, validates only by passing
non-empty strings, and forwards them to the worker for both dry-run and
`--send`. The runbook then uses scoped CLI commands for the target proof row
instead of requiring all other paid delivery rows to be absent.

## Intentional

- No scheduler metadata override or scoped scheduler mode. Scheduled delivery
  should remain queue-wide; this slice scopes only the manual proof CLI and
  shared worker entry point.
- No live database, Stripe, portfolio, or Resend call is executed in this PR.
  #1921 PR 4 still owns the actual deployed proof once operator inputs exist.
- No request-id format validation in the CLI. The database query is parameterized
  and no looser than the existing row identity; invalid IDs simply match no rows.

## Deferred

- #1921 PR 4 still owns the deployed live proof artifact and tracker closeout.
- If the operator wants scheduler-level one-shot scoped execution later, add a
  separate task metadata contract rather than overloading this manual CLI slice.

Parked hardening: none.

## Verification

- Command: python -m py_compile atlas_brain/content_ops_deflection_delivery.py scripts/send_content_ops_deflection_report_deliveries.py tests/test_atlas_content_ops_deflection_delivery.py tests/test_send_content_ops_deflection_report_deliveries.py tests/test_content_ops_deflection_launch_preflight_runbook.py - passed.
- Command: pytest tests/test_atlas_content_ops_deflection_delivery.py tests/test_send_content_ops_deflection_report_deliveries.py tests/test_deflection_report_delivery_task.py tests/test_content_ops_deflection_launch_preflight_runbook.py -q - 62 passed, 1 skipped, 1 warning.
- Command: python scripts/audit_extracted_pipeline_ci_enrollment.py - OK: 195 matching tests are enrolled.
- Pending before push:
  - Command: python scripts/sync_pr_plan.py plans/PR-Deflection-Scoped-Delivery-Proof.md --check
  - Command: bash scripts/local_pr_review.sh

## Estimated diff size

| File | LOC |
|---|---:|
| `atlas_brain/content_ops_deflection_delivery.py` | 18 |
| `docs/extraction/validation/content_ops_deflection_launch_preflight_runbook.md` | 67 |
| `plans/PR-Deflection-Scoped-Delivery-Proof.md` | 123 |
| `scripts/send_content_ops_deflection_report_deliveries.py` | 20 |
| `tests/test_atlas_content_ops_deflection_delivery.py` | 74 |
| `tests/test_content_ops_deflection_launch_preflight_runbook.py` | 32 |
| `tests/test_send_content_ops_deflection_report_deliveries.py` | 61 |
| **Total** | **395** |

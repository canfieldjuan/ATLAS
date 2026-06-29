# PR-Billing-Checkout-Missing-Report-Live-Adapter

## Why this slice exists

The billing real-adapters lane is replacing hand-written fake DB-pool Stripe
money-path tests with live asyncpg/Postgres state assertions. After the
checkout-completion paid and no-delivery-email paths moved to live coverage, the
paid-but-missing-report branch still uses `_Pool.update_result = "UPDATE 0"` and
SQL-string assertions to prove the direct handler fails closed with 409 when a
Checkout payment arrives before the report row exists.

Root cause: `test_deflection_checkout_completion_fails_closed_when_report_missing`
asserts fake update results and SQL call arguments instead of proving the real
asyncpg/Postgres adapter observes the missing report row, raises the retryable
409, and queues no delivery. This PR fixes the root for that one handler path by
running `_handle_content_ops_deflection_report_checkout_completed` against a
migrated Postgres database with no deflection report row for the paid account /
request.

## Scope (this PR)

Ownership lane: real-adapters/test-quality
Slice phase: Production hardening

1. Convert `test_deflection_checkout_completion_fails_closed_when_report_missing`
   from `_Pool`/SQL-call assertions to the live asyncpg/Postgres harness.
2. Preserve the behavior contract: checkout completion raises HTTP 409 when the
   paid report row is absent, so Stripe retries instead of silently accepting a
   paid-but-undeliverable report.
3. Prove no delivery row is queued for the account when the report is missing.
4. Keep maturity-sweep honest: only ratchet `atlas_brain/api/billing.py` if this
   conversion earns a detector-visible reduction. Remaining `_Pool` tests stay
   grep-visible for later burn-down slices.

### Review Contract

Acceptance criteria:

- The converted test uses the real asyncpg pool wrapper, applies live billing
  migrations, cleans any existing rows for the generated account/request, and
  closes the pool in `finally`.
- The test calls the real `_handle_content_ops_deflection_report_checkout_completed`
  handler, not the full webhook, because this slice is replacing the direct
  handler fake-pool proof.
- No deflection report row is seeded for the account/request under test.
- The test asserts HTTP 409 from the handler and the `"Deflection report not found"`
  detail.
- State is asserted from Postgres: no report row exists and no delivery row
  exists for the account at account scope.
- Remaining `_Pool` tests stay detector-visible for later slices.

Affected surfaces:

- `tests/test_atlas_billing_content_ops_deflection_stripe_paid.py` only.
- Runtime billing code is exercised through the existing direct handler, but
  production implementation files are intentionally unchanged.

Risk areas:

- Billing checkout completion when Stripe confirms payment before the report row
  is committed.
- The fail-closed path must remain retryable and must not queue delivery for a
  missing report.
- The live adapter proof must not weaken the old fake-pool missing-report
  contract.

Reviewer rules: R1, R2, R3, R6, R8, R10, R13, R14.

### Files touched

- `plans/PR-Billing-Checkout-Missing-Report-Live-Adapter.md`
- `tests/test_atlas_billing_content_ops_deflection_stripe_paid.py`

## Mechanism

Use the existing live billing pool and migration helpers, clean the generated
account/request, and intentionally do not call `_seed_live_deflection_report`.
Call `_handle_content_ops_deflection_report_checkout_completed` with the
existing synthetic Stripe Checkout session. Assert the handler raises
`billing.HTTPException` with status 409, then query
`content_ops_deflection_reports` and `content_ops_deflection_report_deliveries`
to prove the missing-report branch left no report or delivery state behind.

## Intentional

- This is one direct-handler fake-pool burn-down, not the whole checkout missing
  report / reconciliation cluster.
- This slice keeps the unknown-event-age 409 behavior covered. The aged-event
  reconciliation branch is deferred because it needs the reconciliation ledger
  migrations and a separate live-state contract.
- This slice does not insert or assert `billing_events`; the direct handler does
  not own event dedupe/audit logging.
- The Stripe session remains synthetic; the database adapter and persisted state
  are real.

## Deferred

- Live conversion of the aged paid-but-missing reconciliation tests is deferred
  to a follow-up slice that includes migrations
  `336_content_ops_deflection_paid_reconciliation.sql` and
  `337_content_ops_deflection_reconciliation_null_session.sql`.
- Remaining billing fake-pool checkout/refund/dispute/delta tests and the
  temporary `_resolve_billing_db_pool` direct-call shim are deferred to follow-up
  real-adapter burn-down slices.

Parked hardening: none.

## Verification

- Python compile check for
  `tests/test_atlas_billing_content_ops_deflection_stripe_paid.py` with
  `python -m py_compile` - passed.
- `ATLAS_MIGRATION_TEST_DATABASE_URL=postgresql://atlas:atlas_dev_password@localhost:5433/atlas python -m pytest tests/test_atlas_billing_content_ops_deflection_stripe_paid.py::test_deflection_checkout_completion_fails_closed_when_report_missing -q` - passed, 1 test.
- `python -m pytest tests/test_atlas_billing_content_ops_deflection_stripe_paid.py -q` - passed, 40 passed / 19 skipped.
- `ATLAS_MIGRATION_TEST_DATABASE_URL=postgresql://atlas:atlas_dev_password@localhost:5433/atlas python -m pytest tests/test_atlas_billing_content_ops_deflection_stripe_paid.py -q` - passed, 59 passed.
- `python scripts/maturity_sweep.py atlas_brain/api --tests-root tests --baseline tests/maturity_sweep/baseline_atlas_brain_api.json --top 80` - passed; no baseline update because `billing.py` remained `INTERNAL_MOCK x38`, score 178.

## Estimated diff size

| File | LOC |
|---|---:|
| `plans/PR-Billing-Checkout-Missing-Report-Live-Adapter.md` | 124 |
| `tests/test_atlas_billing_content_ops_deflection_stripe_paid.py` | 67 |
| **Total** | **191** |

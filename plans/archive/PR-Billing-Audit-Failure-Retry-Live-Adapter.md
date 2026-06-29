# PR-Billing-Audit-Failure-Retry-Live-Adapter

## Why this slice exists

The billing real-adapters lane is replacing hand-written fake DB-pool webhook
tests with live asyncpg/Postgres state assertions one money-path boundary at a
time. #1899 covered duplicate checkout idempotency. The checkout audit-failure
retry path still proves "side effect can complete before the billing audit row,
then a Stripe retry restores idempotency" with `_Pool.fail_billing_event_insert`,
`processed_event_ids`, and fake `execute_calls`.

Root cause:
`test_stripe_webhook_retry_after_audit_failure_restores_idempotency` simulates a
`billing_events` insert failure in a fake pool after mutating fake report and
delivery dictionaries. That does not prove the real asyncpg/Postgres adapter
preserves the paid report unlock when the audit insert fails, leaves the event
undeduped for a Stripe retry, then inserts the dedupe row on retry so the next
delivery is idempotently skipped. This PR fixes the root for that one retry path
by running the real webhook against migrated Postgres and forcing the first
audit insert to fail at the database layer.

## Scope (this PR)

Ownership lane: real-adapters/test-quality
Slice phase: Production hardening

1. Convert
   `test_stripe_webhook_retry_after_audit_failure_restores_idempotency` from
   `_Pool`/SQL-call assertions to the live asyncpg/Postgres harness.
2. Preserve the current behavior contract: first webhook returns `ok` and marks
   the report paid even when the `billing_events` insert fails; the event is not
   deduped; a retry returns `ok`, keeps the report paid, and inserts the
   `billing_events` row; a later duplicate returns `already_processed` without
   queuing another delivery.
3. Keep maturity-sweep honest: only ratchet `atlas_brain/api/billing.py` if this
   conversion earns a detector-visible reduction. Remaining `_Pool` tests stay
   grep-visible for later burn-down slices.

### Review Contract

Acceptance criteria:

- The converted test uses the real asyncpg pool wrapper, applies live billing
  migrations, seeds `saas_accounts` plus an unpaid deflection report row, and
  cleans up all rows plus any test trigger/function in `finally`.
- The first audit insert failure is produced by Postgres itself for the specific
  `stripe_event_id`, not by a fake pool, fake call list, or query-string
  assertion.
- Stripe remains mocked only at the external webhook SDK boundary; report,
  delivery, and `billing_events` state are asserted from Postgres.
- The test asserts first-run side effects survive the audit failure, the audit
  row is absent after that failure, retry inserts exactly one audit row, and the
  third duplicate does not create an additional delivery row.
- Remaining `_Pool` tests stay detector-visible for later slices.

Affected surfaces:

- `tests/test_atlas_billing_content_ops_deflection_stripe_paid.py` only.
- Runtime Stripe webhook code is exercised through the existing real endpoint
  harness, but production implementation files are intentionally unchanged.

Risk areas:

- Billing/webhook idempotency after a money-path audit insert failure.
- A retry after an audit failure must restore idempotency without double
  delivery.
- The live adapter proof must not weaken the old fake-pool retry contract.

Reviewer rules: R1, R2, R3, R6, R8, R10, R13, R14.

### Files touched

- `plans/PR-Billing-Audit-Failure-Retry-Live-Adapter.md`
- `tests/maturity_sweep/baseline_atlas_brain_api.json`
- `tests/test_atlas_billing_content_ops_deflection_stripe_paid.py`

## Mechanism

Seed a live unpaid report through `PostgresDeflectionReportArtifactStore`, then
install a uniquely named Postgres trigger/function that raises before inserting
the matching `billing_events.stripe_event_id`. Run `_run_stripe_webhook` once and
assert the response is `ok`, the incident log records the audit insert failure,
the report is paid, the delivery row is pending, and the billing-event count is
zero.

Drop the trigger, run the same webhook again, and assert the retry returns `ok`
with the report and delivery still in the paid/pending state and exactly one
`billing_events` row. Run a third duplicate webhook and assert it returns
`already_processed` and the account still has exactly one delivery row.

## Intentional

- This is one fake-pool burn-down, not the whole audit-failure/idempotency
  cluster.
- The Stripe webhook object remains a fake SDK boundary; the DB state and the
  forced audit-insert failure use the real Postgres adapter.
- The Postgres trigger is per-test and keyed to one synthetic `stripe_event_id`
  so it cannot affect other tests or rows; cleanup drops both trigger and
  function.

## Deferred

- `test_stripe_webhook_keeps_paid_unlock_when_audit_insert_fails` still covers
  the one-shot audit-failure log shape with `_Pool`; it is mostly subsumed by
  this retry proof but remains grep-visible for a later consolidation slice.
- Remaining billing fake-pool checkout/refund/dispute tests and the temporary
  `_resolve_billing_db_pool` direct-call shim are deferred to follow-up
  real-adapter burn-down slices.

Parked hardening: none.

## Verification

- Python compile check for
  `tests/test_atlas_billing_content_ops_deflection_stripe_paid.py` with
  `python -m py_compile` - passed.
- `ATLAS_MIGRATION_TEST_DATABASE_URL=postgresql://atlas:atlas_dev_password@localhost:5433/atlas python -m pytest tests/test_atlas_billing_content_ops_deflection_stripe_paid.py::test_stripe_webhook_retry_after_audit_failure_restores_idempotency -q` - passed, 1 test.
- `python -m pytest tests/test_atlas_billing_content_ops_deflection_stripe_paid.py -q` - passed, 44 passed / 14 skipped.
- `ATLAS_MIGRATION_TEST_DATABASE_URL=postgresql://atlas:atlas_dev_password@localhost:5433/atlas python -m pytest tests/test_atlas_billing_content_ops_deflection_stripe_paid.py -q` - passed, 58 passed.
- `python scripts/maturity_sweep.py atlas_brain/api --tests-root tests --baseline tests/maturity_sweep/baseline_atlas_brain_api.json --update-baseline` - passed; updated only `atlas_brain/api/billing.py`.
- `python scripts/maturity_sweep.py atlas_brain/api --tests-root tests --baseline tests/maturity_sweep/baseline_atlas_brain_api.json --top 80` - passed; `billing.py` is now `INTERNAL_MOCK x41`, score 190.

## Estimated diff size

| File | LOC |
|---|---:|
| `plans/PR-Billing-Audit-Failure-Retry-Live-Adapter.md` | 130 |
| `tests/maturity_sweep/baseline_atlas_brain_api.json` | 4 |
| `tests/test_atlas_billing_content_ops_deflection_stripe_paid.py` | 231 |
| **Total** | **365** |

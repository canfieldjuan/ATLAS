# PR-Billing-Audit-Failure-One-Shot-Live-Adapter

## Why this slice exists

The billing real-adapters lane is replacing hand-written fake DB-pool webhook
tests with live asyncpg/Postgres state assertions one money-path boundary at a
time. #1900 converted the audit-failure retry path and deliberately deferred the
older one-shot audit-failure log/unlock test. That test still uses
`_Pool.fail_billing_event_insert`, fake `execute_calls`, and
`processed_event_ids` to prove "paid unlock survives a failed billing audit
insert."

Root cause:
`test_stripe_webhook_keeps_paid_unlock_when_audit_insert_fails` simulates a
`billing_events` insert failure in a fake pool and then asserts fake SQL-call
order. That does not prove the real asyncpg/Postgres adapter keeps a paid
deflection report unlocked and delivery-queued when the audit insert fails. This
PR fixes the root for that one-shot path by reusing the #1900 Postgres trigger
failure mechanism and asserting persisted report, delivery, and audit-row state.

## Scope (this PR)

Ownership lane: real-adapters/test-quality
Slice phase: Production hardening

1. Convert
   `test_stripe_webhook_keeps_paid_unlock_when_audit_insert_fails` from
   `_Pool`/SQL-call assertions to the live asyncpg/Postgres harness.
2. Preserve the current behavior contract: webhook returns `ok`, logs the audit
   insert failure, marks the report paid, queues one delivery, and does not
   record a `billing_events` row while the DB insert is failing.
3. Keep maturity-sweep honest: only ratchet `atlas_brain/api/billing.py` if this
   conversion earns a detector-visible reduction. Remaining `_Pool` tests stay
   grep-visible for later burn-down slices.

### Review Contract

Acceptance criteria:

- The converted test uses the real asyncpg pool wrapper, applies live billing
  migrations, seeds `saas_accounts` plus an unpaid deflection report row, and
  cleans up all rows plus any test trigger/function in `finally`.
- The audit insert failure is produced by Postgres itself for the specific
  `stripe_event_id`, not by a fake pool, fake call list, or query-string
  assertion.
- Stripe remains mocked only at the external webhook SDK boundary; report,
  delivery, and `billing_events` state are asserted from Postgres.
- The test asserts the response, incident log, paid report state, pending
  delivery state, and absent billing-event row after the forced failure.
- Remaining `_Pool` tests stay detector-visible for later slices.

Affected surfaces:

- `tests/test_atlas_billing_content_ops_deflection_stripe_paid.py` only.
- Runtime Stripe webhook code is exercised through the existing real endpoint
  harness, but production implementation files are intentionally unchanged.

Risk areas:

- Billing/webhook fail-open behavior after a money-path audit insert failure.
- A paid customer must not lose access because the audit row failed to write.
- The live adapter proof must not weaken the old fake-pool one-shot contract.

Reviewer rules: R1, R2, R3, R6, R8, R10, R13, R14.

### Files touched

- `plans/PR-Billing-Audit-Failure-One-Shot-Live-Adapter.md`
- `tests/maturity_sweep/baseline_atlas_brain_api.json`
- `tests/test_atlas_billing_content_ops_deflection_stripe_paid.py`

## Mechanism

Seed a live unpaid report through `PostgresDeflectionReportArtifactStore`, then
install a uniquely named Postgres trigger/function that raises before inserting
the matching `billing_events.stripe_event_id`. Run `_run_stripe_webhook` once and
assert the response is `ok`, the incident log records the audit insert failure,
the report is paid with the Stripe session reference, the delivery row is
pending, and the billing-event count for that event remains zero.

## Intentional

- This is one fake-pool burn-down, not the whole checkout/delta/drain fake-pool
  cluster.
- The Stripe webhook object remains a fake SDK boundary; the DB state and the
  forced audit-insert failure use the real Postgres adapter.
- The Postgres trigger helper already landed in #1900; this slice reuses it
  instead of adding another failure-injection path.

## Deferred

- Remaining billing fake-pool checkout/refund/dispute/delta tests and the
  temporary `_resolve_billing_db_pool` direct-call shim are deferred to follow-up
  real-adapter burn-down slices.

Parked hardening: none.

## Verification

- Python compile check for
  `tests/test_atlas_billing_content_ops_deflection_stripe_paid.py` with
  `python -m py_compile` - passed.
- `ATLAS_MIGRATION_TEST_DATABASE_URL=postgresql://atlas:atlas_dev_password@localhost:5433/atlas python -m pytest tests/test_atlas_billing_content_ops_deflection_stripe_paid.py::test_stripe_webhook_keeps_paid_unlock_when_audit_insert_fails -q` - passed, 1 test.
- `python -m pytest tests/test_atlas_billing_content_ops_deflection_stripe_paid.py -q` - passed, 43 passed / 15 skipped.
- `ATLAS_MIGRATION_TEST_DATABASE_URL=postgresql://atlas:atlas_dev_password@localhost:5433/atlas python -m pytest tests/test_atlas_billing_content_ops_deflection_stripe_paid.py -q` - passed, 58 passed.
- `python scripts/maturity_sweep.py atlas_brain/api --tests-root tests --baseline tests/maturity_sweep/baseline_atlas_brain_api.json --update-baseline` - passed; updated only `atlas_brain/api/billing.py`.
- `python scripts/maturity_sweep.py atlas_brain/api --tests-root tests --baseline tests/maturity_sweep/baseline_atlas_brain_api.json --top 80` - passed; `billing.py` is now `INTERNAL_MOCK x38`, score 178.

## Estimated diff size

| File | LOC |
|---|---:|
| `plans/PR-Billing-Audit-Failure-One-Shot-Live-Adapter.md` | 116 |
| `tests/maturity_sweep/baseline_atlas_brain_api.json` | 4 |
| `tests/test_atlas_billing_content_ops_deflection_stripe_paid.py` | 134 |
| **Total** | **254** |

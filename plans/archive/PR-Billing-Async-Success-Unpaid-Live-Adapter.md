# PR-Billing-Async-Success-Unpaid-Live-Adapter

## Why this slice exists

The billing real-adapters lane is replacing fake DB-pool webhook tests with
live asyncpg/Postgres state assertions one money-path boundary at a time. #1888
proved `checkout.session.async_payment_succeeded` unlocks a report when Stripe
marks the session paid; #1890 proved completed-but-unpaid stays locked. The
async-success event with `payment_status="unpaid"` still relies on the
hand-written `_Pool` fake and SQL-call assertions.

Root cause: the async-success unpaid test verifies fake `execute_calls`,
`report_rows`, and `delivery_rows`. That does not prove the real webhook records
the async-success event without unlocking the report when the session has not
actually been paid. This PR fixes that root for one more fail-closed payment
boundary.

## Scope (this PR)

Ownership lane: real-adapters/test-quality
Slice phase: Production hardening

1. Convert
   `test_stripe_webhook_deflection_async_success_unpaid_stays_pending` from
   `_Pool`/SQL-call assertions to the live asyncpg/Postgres harness.
2. Reuse the live locked-state helper to assert real persisted state: report
   remains unpaid, no payment reference is written, no delivery row is queued,
   and the billing event is recorded once.
3. Ratchet `atlas_brain/api/billing.py` INTERNAL_MOCK only by the earned count
   from the removed fake-pool test.

### Review Contract

- The converted test uses the real asyncpg pool wrapper, applies the live
  billing migrations, seeds `saas_accounts` and a real deflection report row,
  and cleans up in `finally`.
- The test asserts real persisted effects: event recorded, report remains
  locked, delivery remains absent account-wide, and duplicate webhook handling
  is idempotent.
- Stripe remains mocked only at the external SDK/webhook-construction boundary.
- Remaining `_Pool` tests stay detector-visible for later slices.

### Files touched

- `plans/PR-Billing-Async-Success-Unpaid-Live-Adapter.md`
- `tests/maturity_sweep/baseline_atlas_brain_api.json`
- `tests/test_atlas_billing_content_ops_deflection_stripe_paid.py`

## Mechanism

Build the existing unpaid async-success Checkout Session event with real
`account_id`, `request_id`, and `session_id` values. Seed a real report row via
`PostgresDeflectionReportArtifactStore`, run the actual `billing.stripe_webhook`
entrypoint through `_run_stripe_webhook`, and verify the persisted locked state
with `_assert_live_deflection_report_locked`, including account-wide absence of
delivery rows. Then run the same webhook a second time to prove idempotency
against the real `billing_events` table.

## Intentional

- This is one fake-pool burn-down, not a broad webhook-test sweep.
- The existing live locked-state helper is reused instead of adding another
  assertion shape.

## Deferred

- Remaining billing fake-pool webhook tests and the temporary
  `_resolve_billing_db_pool` direct-call shim are deferred to follow-up
  real-adapter burn-down slices.

Parked hardening: none.

## Verification

- Python compile check for `tests/test_atlas_billing_content_ops_deflection_stripe_paid.py` - passed.
- `ATLAS_MIGRATION_TEST_DATABASE_URL=postgresql://atlas:atlas_dev_password@localhost:5433/atlas python -m pytest tests/test_atlas_billing_content_ops_deflection_stripe_paid.py::test_stripe_webhook_deflection_async_success_unpaid_stays_pending -q` - passed, 1 test.
- `python -m pytest tests/test_atlas_billing_content_ops_deflection_stripe_paid.py -q` - passed, 53 passed / 5 skipped.
- `ATLAS_MIGRATION_TEST_DATABASE_URL=postgresql://atlas:atlas_dev_password@localhost:5433/atlas python -m pytest tests/test_atlas_billing_content_ops_deflection_stripe_paid.py -q` - passed, 58 passed.
- `python scripts/maturity_sweep.py atlas_brain/api --tests-root tests --baseline tests/maturity_sweep/baseline_atlas_brain_api.json --top 80` - passed; `billing.py` is now `INTERNAL_MOCK x47`, score 214.
- Review-fix verification: `ATLAS_MIGRATION_TEST_DATABASE_URL=postgresql://atlas:atlas_dev_password@localhost:5433/atlas python -m pytest tests/test_atlas_billing_content_ops_deflection_stripe_paid.py::test_stripe_webhook_deflection_completed_unpaid_stays_pending tests/test_atlas_billing_content_ops_deflection_stripe_paid.py::test_stripe_webhook_deflection_async_success_unpaid_stays_pending tests/test_atlas_billing_content_ops_deflection_stripe_paid.py::test_stripe_webhook_deflection_async_failure_is_observed_without_unlock -q` - passed, 3 locked-path tests.

## Estimated diff size

| File | LOC |
|---|---:|
| `plans/PR-Billing-Async-Success-Unpaid-Live-Adapter.md` | 89 |
| `tests/maturity_sweep/baseline_atlas_brain_api.json` | 4 |
| `tests/test_atlas_billing_content_ops_deflection_stripe_paid.py` | 105 |
| **Total** | **198** |

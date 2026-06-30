# PR-Billing-Checkout-No-Email-Live-Adapter

## Why this slice exists

The billing real-adapters lane is replacing hand-written fake DB-pool Stripe
money-path tests with live asyncpg/Postgres state assertions. After the
checkout-completion paid happy path moved to live coverage, the adjacent
no-delivery-email case still uses `_Pool.execute_calls` and fake
`delivery_rows == {}` to prove that a paid report is unlocked but no delivery is
queued when the report has no buyer email.

Root cause: `test_deflection_checkout_completion_skips_delivery_queue_without_email`
asserts SQL strings and fake in-memory delivery state instead of proving the
real Postgres adapter persists the paid report and skips
`content_ops_deflection_report_deliveries` at account scope. This PR fixes the
root for that one handler path by running the direct checkout-completion handler
against a migrated Postgres database with a real report row whose
`delivery_email` is `NULL`.

## Scope (this PR)

Ownership lane: real-adapters/test-quality
Slice phase: Production hardening

1. Convert `test_deflection_checkout_completion_skips_delivery_queue_without_email`
   from `_Pool`/SQL-call assertions to the live asyncpg/Postgres harness.
2. Preserve the behavior contract: checkout completion still marks the report
   paid with the Stripe session reference, but queues no delivery row when the
   report has no delivery email.
3. Keep maturity-sweep honest: only ratchet `atlas_brain/api/billing.py` if this
   conversion earns a detector-visible reduction. Remaining `_Pool` tests stay
   grep-visible for later burn-down slices.

### Review Contract

Acceptance criteria:

- The converted test uses the real asyncpg pool wrapper, applies live billing
  migrations, seeds `saas_accounts` plus an unpaid deflection report row with
  `delivery_email=None`, and cleans up in `finally`.
- The test calls the real `_handle_content_ops_deflection_report_checkout_completed`
  handler, not the full webhook, because this slice is replacing the direct
  handler fake-pool proof.
- Report and delivery state are asserted from Postgres; no SQL-string or fake
  call-list assertions remain in this test.
- The test asserts the report is paid with the expected `payment_reference`.
- The no-delivery assertion is account-scoped (`COUNT(*) WHERE account_id = $1`)
  so an unexpected/default request id cannot hide a queued delivery.
- Remaining `_Pool` tests stay detector-visible for later slices.

Affected surfaces:

- `tests/test_atlas_billing_content_ops_deflection_stripe_paid.py` only.
- Runtime billing code is exercised through the existing direct handler, but
  production implementation files are intentionally unchanged.

Risk areas:

- Billing checkout completion on the paid report unlock path.
- The missing-delivery-email branch must not enqueue a delivery under any
  request id for the account.
- The live adapter proof must not weaken the old fake-pool no-delivery contract.

Reviewer rules: R1, R2, R3, R6, R8, R10, R13, R14.

### Files touched

- `plans/PR-Billing-Checkout-No-Email-Live-Adapter.md`
- `tests/test_atlas_billing_content_ops_deflection_stripe_paid.py`

## Mechanism

Add an optional `delivery_email` parameter to the existing live report seed
helper, defaulting to the current `buyer@example.com` behavior so existing live
tests do not change. The converted test seeds a live unpaid report with
`delivery_email=None`, calls `_handle_content_ops_deflection_report_checkout_completed`
with the existing synthetic Stripe Checkout session, then queries
`content_ops_deflection_reports` to prove the report was marked paid and
`content_ops_deflection_report_deliveries` to prove no delivery row exists for
that account.

## Intentional

- This is one direct-handler fake-pool burn-down, not the whole checkout helper
  cluster.
- This slice does not insert or assert `billing_events`; the direct handler does
  not own event dedupe/audit logging.
- The Stripe session remains synthetic; the database adapter and persisted state
  are real.
- The seed helper parameter is narrowly scoped to test setup; production report
  creation code is unchanged.

## Deferred

- Remaining billing fake-pool checkout/refund/dispute/delta tests and the
  temporary `_resolve_billing_db_pool` direct-call shim are deferred to follow-up
  real-adapter burn-down slices.

Parked hardening: none.

## Verification

- Python compile check for
  `tests/test_atlas_billing_content_ops_deflection_stripe_paid.py` with
  `python -m py_compile` - passed.
- `ATLAS_MIGRATION_TEST_DATABASE_URL=postgresql://atlas:atlas_dev_password@localhost:5433/atlas python -m pytest tests/test_atlas_billing_content_ops_deflection_stripe_paid.py::test_deflection_checkout_completion_skips_delivery_queue_without_email -q` - passed, 1 test.
- `python -m pytest tests/test_atlas_billing_content_ops_deflection_stripe_paid.py -q` - passed, 41 passed / 18 skipped.
- `ATLAS_MIGRATION_TEST_DATABASE_URL=postgresql://atlas:atlas_dev_password@localhost:5433/atlas python -m pytest tests/test_atlas_billing_content_ops_deflection_stripe_paid.py -q` - passed, 59 passed.
- `python scripts/maturity_sweep.py atlas_brain/api --tests-root tests --baseline tests/maturity_sweep/baseline_atlas_brain_api.json --top 80` - passed; no baseline update because `billing.py` remained `INTERNAL_MOCK x38`, score 178.

## Estimated diff size

| File | LOC |
|---|---:|
| `plans/PR-Billing-Checkout-No-Email-Live-Adapter.md` | 117 |
| `tests/test_atlas_billing_content_ops_deflection_stripe_paid.py` | 74 |
| **Total** | **191** |

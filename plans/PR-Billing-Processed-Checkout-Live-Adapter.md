# PR-Billing-Processed-Checkout-Live-Adapter

## Why this slice exists

The billing real-adapters lane is replacing hand-written fake DB-pool webhook
tests with live asyncpg/Postgres state assertions one money-path boundary at a
time. #1898 covered duplicate refund idempotency. The duplicate deflection
checkout path still proves "return before marking paid" with
`_Pool.processed_event_ids`, fake fetch-call arguments, and `execute_calls == []`.

Root cause: `test_stripe_webhook_skips_processed_deflection_checkout_before_paid_update`
asserts fake dedupe state and fake SQL-call absence instead of a real
`billing_events` dedupe row and persisted report/delivery state. That does not
prove the real webhook short-circuits before marking a deflection report paid
when a checkout event was already processed. This PR fixes the root for that one
processed-checkout path by running the real webhook against a migrated Postgres
database.

## Scope (this PR)

Ownership lane: real-adapters/test-quality
Slice phase: Production hardening

1. Convert
   `test_stripe_webhook_skips_processed_deflection_checkout_before_paid_update`
   from `_Pool`/SQL-call assertions to the live asyncpg/Postgres harness.
2. Preserve the current behavior contract: a duplicate deflection checkout event
   returns `already_processed`, does not mark the report paid, does not create
   delivery rows, and leaves the existing billing event as the only audit row.
3. Keep maturity-sweep honest: do not ratchet
   `atlas_brain/api/billing.py` beyond the earned reduction. Remaining `_Pool`
   tests stay grep-visible for later burn-down slices.

### Review Contract

Acceptance criteria:

- The converted test uses the real asyncpg pool wrapper, applies live billing
  migrations, seeds `saas_accounts`, an unpaid deflection report row, and a
  pre-existing `billing_events` row for the checkout event, then cleans up in
  `finally`.
- Stripe remains mocked only at the external webhook SDK boundary; the test
  exercises the real webhook path until the dedupe row short-circuits before any
  paid-report update.
- The test asserts persisted report state remains unpaid with no payment
  reference, delivery rows remain absent account-wide, and the duplicate
  checkout has exactly one `billing_events` row.
- The test proves the duplicate checkout is a report-row no-op by asserting
  `updated_at` is unchanged across webhook handling.
- Remaining `_Pool` tests stay detector-visible for later slices.

Affected surfaces:

- `tests/test_atlas_billing_content_ops_deflection_stripe_paid.py` only.
- Runtime Stripe webhook code is exercised through the existing real endpoint
  harness, but production implementation files are intentionally unchanged.

Risk areas:

- Billing and webhook idempotency on a money-path checkout event.
- A duplicate checkout must not unlock/report-deliver a paid report twice.
- Live adapter coverage must not weaken the old fake-pool no-write/no-delivery
  contract while removing SQL-call assertions.

Reviewer rules: R1, R2, R3, R6, R8, R10, R13, R14.

### Files touched

- `plans/PR-Billing-Processed-Checkout-Live-Adapter.md`
- `tests/maturity_sweep/baseline_atlas_brain_api.json`
- `tests/test_atlas_billing_content_ops_deflection_stripe_paid.py`

## Mechanism

Seed a live unpaid report through `PostgresDeflectionReportArtifactStore`, insert
a matching `billing_events` row for the checkout event id, and build the existing
`checkout.session.completed` event. Run `_run_stripe_webhook` against the live
pool and query Postgres to prove the webhook returns `already_processed`, the
report remains unpaid and untouched, no delivery rows exist for the account, and
the billing-event count remains one.

## Intentional

- This is one fake-pool burn-down, not the whole idempotency/audit-failure
  cluster.
- The Stripe webhook object remains a fake SDK boundary; the DB state uses the
  real adapter.

## Deferred

- Remaining billing fake-pool checkout/refund/dispute tests and the temporary
  `_resolve_billing_db_pool` direct-call shim are deferred to follow-up
  real-adapter burn-down slices.

Parked hardening: none.

## Verification

- Python compile check for `tests/test_atlas_billing_content_ops_deflection_stripe_paid.py` with `python -m py_compile` - passed.
- `ATLAS_MIGRATION_TEST_DATABASE_URL=postgresql://atlas:atlas_dev_password@localhost:5433/atlas python -m pytest tests/test_atlas_billing_content_ops_deflection_stripe_paid.py::test_stripe_webhook_skips_processed_deflection_checkout_before_paid_update -q` - passed, 1 test.
- `python -m pytest tests/test_atlas_billing_content_ops_deflection_stripe_paid.py -q` - passed, 45 passed / 13 skipped.
- `ATLAS_MIGRATION_TEST_DATABASE_URL=postgresql://atlas:atlas_dev_password@localhost:5433/atlas python -m pytest tests/test_atlas_billing_content_ops_deflection_stripe_paid.py -q` - passed, 58 passed.
- `python scripts/maturity_sweep.py atlas_brain/api --tests-root tests --baseline tests/maturity_sweep/baseline_atlas_brain_api.json --update-baseline` - passed; updated only `atlas_brain/api/billing.py`.
- `python scripts/maturity_sweep.py atlas_brain/api --tests-root tests --baseline tests/maturity_sweep/baseline_atlas_brain_api.json --top 80` - passed; `billing.py` is now `INTERNAL_MOCK x44`, score 202.

## Estimated diff size

| File | LOC |
|---|---:|
| `plans/PR-Billing-Processed-Checkout-Live-Adapter.md` | 113 |
| `tests/maturity_sweep/baseline_atlas_brain_api.json` | 4 |
| `tests/test_atlas_billing_content_ops_deflection_stripe_paid.py` | 118 |
| **Total** | **235** |

# PR-Billing-Processed-Refund-Live-Adapter

## Why this slice exists

The billing real-adapters lane is replacing hand-written fake DB-pool webhook
tests with live asyncpg/Postgres state assertions one money-path boundary at a
time. #1897 covered the unmapped-refund no-mutation branch. The processed-refund
idempotency branch still proves "return before revocation" with
`_Pool.processed_event_ids`, fake report rows, and `execute_calls == []`.

Root cause: `test_stripe_webhook_skips_processed_refund_before_revocation`
asserts fake dedupe state and fake SQL-call absence instead of a real
`billing_events` dedupe row and persisted report/delivery state. That does not
prove the real webhook short-circuits before Stripe lookup or refund revocation
when a refund event was already processed. This PR fixes the root for that one
processed-refund path by running the real webhook against a migrated Postgres
database.

## Scope (this PR)

Ownership lane: real-adapters/test-quality
Slice phase: Production hardening

1. Convert
   `test_stripe_webhook_skips_processed_refund_before_revocation` from
   `_Pool`/SQL-call assertions to the live asyncpg/Postgres harness.
2. Preserve the current behavior contract: a duplicate refund event returns
   `already_processed`, does not call Stripe Checkout Session lookup, does not
   revoke or mutate the paid report, does not create delivery rows, and leaves
   the existing billing event as the only audit row.
3. Keep maturity-sweep honest: do not ratchet
   `atlas_brain/api/billing.py` unless the detector reports an earned
   reduction. Remaining `_Pool` tests stay grep-visible for later burn-down
   slices.

### Review Contract

Acceptance criteria:

- The converted test uses the real asyncpg pool wrapper, applies live billing
  migrations, seeds `saas_accounts`, a paid deflection report row, and a
  pre-existing `billing_events` row for the refund event, then cleans up in
  `finally`.
- Stripe remains mocked only at the external webhook/checkout-session SDK
  boundary; the test asserts the checkout-session lookup is not called because
  the dedupe row short-circuits first.
- The test asserts persisted report state remains paid with its payment
  reference preserved, delivery rows remain absent account-wide, and the
  duplicate refund has exactly one `billing_events` row.
- The test proves the duplicate refund is a report-row no-op by asserting
  `updated_at` is unchanged across webhook handling.
- Remaining `_Pool` tests stay detector-visible for later slices.

Affected surfaces:

- `tests/test_atlas_billing_content_ops_deflection_stripe_paid.py` only.
- Runtime Stripe webhook code is exercised through the existing real endpoint
  harness, but production implementation files are intentionally unchanged.

Risk areas:

- Billing and webhook idempotency on a money-path event.
- A duplicate refund must not call Stripe lookup or revoke/mutate paid report
  access.
- Live adapter coverage must not weaken the old fake-pool no-write/no-delivery
  contract while removing SQL-call assertions.

Reviewer rules: R1, R2, R3, R6, R8, R10, R13, R14.

### Files touched

- `plans/PR-Billing-Processed-Refund-Live-Adapter.md`
- `tests/test_atlas_billing_content_ops_deflection_stripe_paid.py`

## Mechanism

Seed a live paid report through `PostgresDeflectionReportArtifactStore`, insert a
matching `billing_events` row for the refund event id, and build the existing
`charge.refunded` event. Run `_run_stripe_webhook` against the live pool and
query Postgres to prove the webhook returns `already_processed`, Stripe lookup
was not called, the report remains paid and untouched, no delivery rows exist
for the account, and the billing-event count remains one.

## Intentional

- This is one fake-pool burn-down, not the whole idempotency cluster.
- Checkout-session lookup remains a fake Stripe SDK boundary because the
  behavior under test is that duplicate events return before any Stripe lookup.

## Deferred

- Remaining billing fake-pool checkout/refund/dispute tests and the temporary
  `_resolve_billing_db_pool` direct-call shim are deferred to follow-up
  real-adapter burn-down slices.

Parked hardening: none.

## Verification

- Python compile check for `tests/test_atlas_billing_content_ops_deflection_stripe_paid.py` with `python -m py_compile` - passed.
- `ATLAS_MIGRATION_TEST_DATABASE_URL=postgresql://atlas:atlas_dev_password@localhost:5433/atlas python -m pytest tests/test_atlas_billing_content_ops_deflection_stripe_paid.py::test_stripe_webhook_skips_processed_refund_before_revocation -q` - passed, 1 test.
- `python -m pytest tests/test_atlas_billing_content_ops_deflection_stripe_paid.py -q` - passed, 46 passed / 12 skipped.
- `ATLAS_MIGRATION_TEST_DATABASE_URL=postgresql://atlas:atlas_dev_password@localhost:5433/atlas python -m pytest tests/test_atlas_billing_content_ops_deflection_stripe_paid.py -q` - passed, 58 passed.
- `python scripts/maturity_sweep.py atlas_brain/api --tests-root tests --baseline tests/maturity_sweep/baseline_atlas_brain_api.json --top 80` - passed; `billing.py` remains `INTERNAL_MOCK x47`, score 214, so no baseline change was earned.

## Estimated diff size

| File | LOC |
|---|---:|
| `plans/PR-Billing-Processed-Refund-Live-Adapter.md` | 112 |
| `tests/test_atlas_billing_content_ops_deflection_stripe_paid.py` | 111 |
| **Total** | **223** |

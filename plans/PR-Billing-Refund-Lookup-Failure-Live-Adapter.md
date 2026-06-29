# PR-Billing-Refund-Lookup-Failure-Live-Adapter

## Why this slice exists

The billing real-adapters lane is replacing hand-written fake DB-pool webhook
tests with live asyncpg/Postgres state assertions one money-path boundary at a
time. #1892 and #1893 converted the successful refund/dispute relock paths. The
refund lookup failure path still proves retryability by checking `_Pool`
in-memory state and `execute_calls == []`.

Root cause: `test_stripe_webhook_refund_lookup_failure_retries_without_unlocking`
uses a fake pool to assert that a Stripe Checkout Session lookup outage raises
503 without recording the webhook event or mutating the paid report. That does
not prove the real database state remains retryable. This PR fixes the root for
that one failure path by running the real webhook against a migrated Postgres
database and asserting the paid report and `billing_events` table directly.

## Scope (this PR)

Ownership lane: real-adapters/test-quality
Slice phase: Production hardening

1. Convert
   `test_stripe_webhook_refund_lookup_failure_retries_without_unlocking` from
   `_Pool`/SQL-call assertions to the live asyncpg/Postgres harness.
2. Preserve the current behavior contract: Stripe checkout-session lookup is
   attempted, lookup failure raises HTTP 503, the report remains paid with its
   original payment reference, no delivery row appears for the account, and no
   `billing_events` row is recorded so the event can retry.
3. Keep maturity-sweep honest: do not ratchet
   `atlas_brain/api/billing.py` unless the detector reports an earned
   reduction. Remaining `_Pool` tests stay grep-visible for later burn-down
   slices.

### Review Contract

- The converted test uses the real asyncpg pool wrapper, applies live billing
  migrations, seeds `saas_accounts` and a paid deflection report row, and cleans
  up in `finally`.
- Stripe remains mocked only at the external webhook/checkout-session SDK
  boundary; the test still forces `checkout.Session.list` to raise and asserts
  the lookup arguments.
- The test asserts persisted report state (`paid=true`, original
  `payment_reference`, `paid_at` still present), account-wide absence of
  delivery rows, and zero `billing_events` rows for the failed event.
- Remaining `_Pool` tests stay detector-visible for later slices.

### Files touched

- `plans/PR-Billing-Refund-Lookup-Failure-Live-Adapter.md`
- `tests/test_atlas_billing_content_ops_deflection_stripe_paid.py`

## Mechanism

Seed a live report through `PostgresDeflectionReportArtifactStore`, mark it paid
with the same payment reference the fake used, and build the existing full
`charge.refunded` event whose Checkout Session lookup raises. Run
`_run_stripe_webhook` against the live pool and assert it raises the expected
503. Then query Postgres to prove the report remains paid, no delivery rows were
created for the account, and the failed event was not inserted into
`billing_events`, preserving retryability.

## Intentional

- This is one fake-pool burn-down, not the whole refund/dispute cluster.
- The checkout-session list remains a fake Stripe SDK boundary because the
  behavior under test is our webhook's DB behavior when Stripe lookup fails.

## Deferred

- Remaining billing fake-pool refund/dispute tests and the temporary
  `_resolve_billing_db_pool` direct-call shim are deferred to follow-up
  real-adapter burn-down slices.

Parked hardening: none.

## Verification

- Python compile check for `tests/test_atlas_billing_content_ops_deflection_stripe_paid.py` with `python -m py_compile` - passed.
- `ATLAS_MIGRATION_TEST_DATABASE_URL=postgresql://atlas:atlas_dev_password@localhost:5433/atlas python -m pytest tests/test_atlas_billing_content_ops_deflection_stripe_paid.py::test_stripe_webhook_refund_lookup_failure_retries_without_unlocking -q` - passed, 1 test.
- `python -m pytest tests/test_atlas_billing_content_ops_deflection_stripe_paid.py -q` - passed, 50 passed / 8 skipped.
- `ATLAS_MIGRATION_TEST_DATABASE_URL=postgresql://atlas:atlas_dev_password@localhost:5433/atlas python -m pytest tests/test_atlas_billing_content_ops_deflection_stripe_paid.py -q` - passed, 58 passed.
- `python scripts/maturity_sweep.py atlas_brain/api --tests-root tests --baseline tests/maturity_sweep/baseline_atlas_brain_api.json --top 80` - passed; `billing.py` remains `INTERNAL_MOCK x47`, score 214, so no baseline change was earned.

## Estimated diff size

| File | LOC |
|---|---:|
| `plans/PR-Billing-Refund-Lookup-Failure-Live-Adapter.md` | 91 |
| `tests/test_atlas_billing_content_ops_deflection_stripe_paid.py` | 89 |
| **Total** | **180** |

# PR-Billing-Refund-Relock-Live-Adapter

## Why this slice exists

The billing real-adapters lane is burning down hand-written fake DB-pool
webhook tests one money-path boundary at a time. #1888, #1889, #1890, and
#1891 moved the checkout completed/async success/async failure paid and unpaid
paths onto live asyncpg/Postgres state checks. The full-refund path still proves
the relock behavior by sniffing fake SQL strings and in-memory dictionaries.

Root cause: `test_stripe_webhook_refund_relocks_paid_deflection_report_via_checkout_lookup`
asserts `_Pool.execute_calls`, `_Pool.report_rows`, and `_Pool.delivery_rows`
instead of the real persisted billing/report/delivery state. That can miss DB
adapter, SQL, and migration drift on a payment-reversal path. This PR fixes the
root for that one refund/relock path by exercising the real webhook entrypoint
against a migrated Postgres database.

## Scope (this PR)

Ownership lane: real-adapters/test-quality
Slice phase: Production hardening

1. Convert
   `test_stripe_webhook_refund_relocks_paid_deflection_report_via_checkout_lookup`
   from `_Pool`/SQL-call assertions to the live asyncpg/Postgres harness.
2. Assert real persisted effects after a full `charge.refunded`: Stripe checkout
   session lookup is called, the paid report is relocked, the existing delivery
   row is revoked, the billing event is recorded exactly once, and duplicate
   webhook handling is idempotent.
3. Keep maturity-sweep honest: do not ratchet
   `atlas_brain/api/billing.py` unless the detector reports an earned
   reduction. Remaining `_Pool` tests stay grep-visible for later burn-down
   slices.

### Review Contract

- The converted test uses the real asyncpg pool wrapper, applies the live
  billing migrations, seeds `saas_accounts`, a real deflection report row, and
  a pre-existing pending delivery row, then cleans up in `finally`.
- Stripe remains mocked only at the external webhook/checkout-session SDK
  boundary; the webhook under test still performs the real checkout-session
  lookup path.
- The test asserts persisted report state (`paid=false`, payment reference
  preserved), persisted delivery state (`delivery_status=revoked`,
  `payment_revoked:charge.refunded` error), billing event idempotency, and the
  expected refund log line.
- Remaining `_Pool` tests stay detector-visible for later slices.

### Files touched

- `plans/PR-Billing-Refund-Relock-Live-Adapter.md`
- `tests/test_atlas_billing_content_ops_deflection_stripe_paid.py`

## Mechanism

Seed a live report through `PostgresDeflectionReportArtifactStore`, mark it
paid with the same payment reference the fake used, and insert the pending
delivery row that the refund path should revoke. Build the existing
`charge.refunded` event with a full-refund charge and a fake Stripe
`checkout.Session.list` response, run `_run_stripe_webhook` against the live
pool, and query Postgres for the report, delivery, and `billing_events` rows.
Then replay the same webhook once to prove the real `billing_events` table
short-circuits duplicates.

## Intentional

- This is one fake-pool burn-down, not the whole refund/dispute cluster.
- The checkout-session list remains a fake Stripe SDK boundary because the
  behavior under test is our webhook's DB side effects after Stripe provides the
  matching Checkout Session.

## Deferred

- Remaining billing fake-pool refund/dispute tests and the temporary
  `_resolve_billing_db_pool` direct-call shim are deferred to follow-up
  real-adapter burn-down slices.

Parked hardening: none.

## Verification

- Python compile check for `tests/test_atlas_billing_content_ops_deflection_stripe_paid.py` with `python -m py_compile` - passed.
- `ATLAS_MIGRATION_TEST_DATABASE_URL=postgresql://atlas:atlas_dev_password@localhost:5433/atlas python -m pytest tests/test_atlas_billing_content_ops_deflection_stripe_paid.py::test_stripe_webhook_refund_relocks_paid_deflection_report_via_checkout_lookup -q` - passed, 1 test.
- `python -m pytest tests/test_atlas_billing_content_ops_deflection_stripe_paid.py -q` - passed, 52 passed / 6 skipped.
- `ATLAS_MIGRATION_TEST_DATABASE_URL=postgresql://atlas:atlas_dev_password@localhost:5433/atlas python -m pytest tests/test_atlas_billing_content_ops_deflection_stripe_paid.py -q` - passed, 58 passed.
- `python scripts/maturity_sweep.py atlas_brain/api --tests-root tests --baseline tests/maturity_sweep/baseline_atlas_brain_api.json --top 80` - passed; `billing.py` remains `INTERNAL_MOCK x47`, score 214, so no baseline change was earned.

## Estimated diff size

| File | LOC |
|---|---:|
| `plans/PR-Billing-Refund-Relock-Live-Adapter.md` | 94 |
| `tests/test_atlas_billing_content_ops_deflection_stripe_paid.py` | 156 |
| **Total** | **250** |

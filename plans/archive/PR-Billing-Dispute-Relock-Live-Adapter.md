# PR-Billing-Dispute-Relock-Live-Adapter

## Why this slice exists

The billing real-adapters lane is replacing hand-written fake DB-pool webhook
tests with live asyncpg/Postgres state assertions one money-path boundary at a
time. #1892 converted full-refund relock/revoke. The adjacent
`charge.dispute.created` relock path still proves the behavior with `_Pool`
SQL-call assertions and in-memory dictionaries.

Root cause: `test_stripe_webhook_dispute_relocks_paid_deflection_report_from_direct_metadata`
asserts `_Pool.execute_calls`, `_Pool.report_rows`, and `_Pool.delivery_rows`
instead of the real persisted report, delivery, and billing-event state. This
can miss SQL, adapter, and migration drift on a chargeback/dispute-created
payment reversal. This PR fixes the root for that one dispute relock path by
running the real webhook against a migrated Postgres database.

## Scope (this PR)

Ownership lane: real-adapters/test-quality
Slice phase: Production hardening

1. Convert
   `test_stripe_webhook_dispute_relocks_paid_deflection_report_from_direct_metadata`
   from `_Pool`/SQL-call assertions to the live asyncpg/Postgres harness.
2. Preserve the current behavior contract: the webhook attempts the Stripe
   Checkout Session lookup by payment intent, falls back to direct dispute
   metadata when no session is found, relocks the paid report, records the
   billing event once, and leaves delivery rows absent account-wide.
3. Keep maturity-sweep honest: do not ratchet
   `atlas_brain/api/billing.py` unless the detector reports an earned
   reduction. Remaining `_Pool` tests stay grep-visible for later burn-down
   slices.

### Review Contract

- The converted test uses the real asyncpg pool wrapper, applies live billing
  migrations, seeds `saas_accounts` and a paid deflection report row, and cleans
  up in `finally`.
- Stripe remains mocked only at the external webhook/checkout-session SDK
  boundary; the test still asserts the checkout-session lookup was attempted
  and returned no matching session before direct metadata was used.
- The test asserts persisted report state (`paid=false`, `paid_at=NULL`,
  payment reference still `NULL`), account-wide absence of delivery rows, and
  real `billing_events` idempotency on duplicate webhook replay.
- Remaining `_Pool` tests stay detector-visible for later slices.

### Files touched

- `plans/PR-Billing-Dispute-Relock-Live-Adapter.md`
- `tests/test_atlas_billing_content_ops_deflection_stripe_paid.py`

## Mechanism

Seed a live report through `PostgresDeflectionReportArtifactStore`, mark it
paid without a payment reference to match the direct-metadata dispute path, and
build the existing `charge.dispute.created` event with source/account/request
metadata on the dispute object. Run `_run_stripe_webhook` against the live pool
with an empty fake Stripe checkout-session list, then query Postgres for the
report, delivery absence, and `billing_events` rows. Replay the same webhook
once to prove the real `billing_events` table short-circuits duplicates.

## Intentional

- This is one fake-pool burn-down, not the whole dispute lifecycle cluster.
- The checkout-session list remains a fake Stripe SDK boundary because the
  behavior under test is our webhook's fallback to direct dispute metadata and
  resulting DB side effects.

## Deferred

- Remaining billing fake-pool refund/dispute tests and the temporary
  `_resolve_billing_db_pool` direct-call shim are deferred to follow-up
  real-adapter burn-down slices.

Parked hardening: none.

## Verification

- Python compile check for `tests/test_atlas_billing_content_ops_deflection_stripe_paid.py` with `python -m py_compile` - passed.
- `ATLAS_MIGRATION_TEST_DATABASE_URL=postgresql://atlas:atlas_dev_password@localhost:5433/atlas python -m pytest tests/test_atlas_billing_content_ops_deflection_stripe_paid.py::test_stripe_webhook_dispute_relocks_paid_deflection_report_from_direct_metadata -q` - passed, 1 test.
- `python -m pytest tests/test_atlas_billing_content_ops_deflection_stripe_paid.py -q` - passed, 51 passed / 7 skipped.
- `ATLAS_MIGRATION_TEST_DATABASE_URL=postgresql://atlas:atlas_dev_password@localhost:5433/atlas python -m pytest tests/test_atlas_billing_content_ops_deflection_stripe_paid.py -q` - passed, 58 passed.
- `python scripts/maturity_sweep.py atlas_brain/api --tests-root tests --baseline tests/maturity_sweep/baseline_atlas_brain_api.json --top 80` - passed; `billing.py` remains `INTERNAL_MOCK x47`, score 214, so no baseline change was earned.

## Estimated diff size

| File | LOC |
|---|---:|
| `plans/PR-Billing-Dispute-Relock-Live-Adapter.md` | 92 |
| `tests/test_atlas_billing_content_ops_deflection_stripe_paid.py` | 97 |
| **Total** | **189** |

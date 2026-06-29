# PR-Billing-Partial-Refund-Live-Adapter

## Why this slice exists

The billing real-adapters lane is replacing hand-written fake DB-pool webhook
tests with live asyncpg/Postgres state assertions one money-path boundary at a
time. #1892 covered full-refund relock, and #1894 covered full-refund lookup
failure retryability. The partial-refund path still proves "observe but do not
revoke" with `_Pool` dictionaries and SQL-call filtering.

Root cause: `test_stripe_webhook_partial_refund_keeps_deflection_report_paid`
asserts fake `report_rows` and `execute_calls` instead of real persisted report,
delivery, and billing-event state. That does not prove the real webhook leaves
paid access intact for a partial refund while still recording the event. This PR
fixes the root for that one non-revocation path by running the real webhook
against a migrated Postgres database.

## Scope (this PR)

Ownership lane: real-adapters/test-quality
Slice phase: Production hardening

1. Convert `test_stripe_webhook_partial_refund_keeps_deflection_report_paid`
   from `_Pool`/SQL-call assertions to the live asyncpg/Postgres harness.
2. Preserve the current behavior contract: partial refund does not call the
   Stripe Checkout Session lookup, does not relock the report, does not create
   or revoke delivery rows, records the billing event once, and logs that the
   partial refund was observed without revocation.
3. Keep maturity-sweep honest: do not ratchet
   `atlas_brain/api/billing.py` unless the detector reports an earned
   reduction. Remaining `_Pool` tests stay grep-visible for later burn-down
   slices.

### Review Contract

- The converted test uses the real asyncpg pool wrapper, applies live billing
  migrations, seeds `saas_accounts` and a paid deflection report row, and cleans
  up in `finally`.
- Stripe remains mocked only at the external webhook/checkout-session SDK
  boundary; the test still asserts no checkout-session lookup happens for a
  partial refund.
- The test asserts persisted report state (`paid=true`, original
  `payment_reference`, `paid_at` still present), account-wide absence of
  delivery rows, and exactly one `billing_events` row for the observed partial
  refund.
- Remaining `_Pool` tests stay detector-visible for later slices.

### Files touched

- `plans/PR-Billing-Partial-Refund-Live-Adapter.md`
- `tests/test_atlas_billing_content_ops_deflection_stripe_paid.py`

## Mechanism

Seed a live report through `PostgresDeflectionReportArtifactStore`, mark it paid
with a stable payment reference, and build the existing partial
`charge.refunded` event (`refunded=false`, `amount_refunded < amount_captured`).
Run `_run_stripe_webhook` against the live pool and query Postgres to prove the
report remains paid, no delivery rows exist for the account, and the webhook
event was still inserted into `billing_events`.

## Intentional

- This is one fake-pool burn-down, not the whole refund/dispute cluster.
- The checkout-session list remains a fake Stripe SDK boundary because the
  behavior under test is that partial refunds return before any lookup.

## Deferred

- Remaining billing fake-pool refund/dispute tests and the temporary
  `_resolve_billing_db_pool` direct-call shim are deferred to follow-up
  real-adapter burn-down slices.

Parked hardening: none.

## Verification

- Python compile check for `tests/test_atlas_billing_content_ops_deflection_stripe_paid.py` with `python -m py_compile` - passed.
- `ATLAS_MIGRATION_TEST_DATABASE_URL=postgresql://atlas:atlas_dev_password@localhost:5433/atlas python -m pytest tests/test_atlas_billing_content_ops_deflection_stripe_paid.py::test_stripe_webhook_partial_refund_keeps_deflection_report_paid -q` - passed, 1 test.
- `python -m pytest tests/test_atlas_billing_content_ops_deflection_stripe_paid.py -q` - passed, 49 passed / 9 skipped.
- `ATLAS_MIGRATION_TEST_DATABASE_URL=postgresql://atlas:atlas_dev_password@localhost:5433/atlas python -m pytest tests/test_atlas_billing_content_ops_deflection_stripe_paid.py -q` - passed, 58 passed.
- `python scripts/maturity_sweep.py atlas_brain/api --tests-root tests --baseline tests/maturity_sweep/baseline_atlas_brain_api.json --top 80` - passed; `billing.py` remains `INTERNAL_MOCK x47`, score 214, so no baseline change was earned.

## Estimated diff size

| File | LOC |
|---|---:|
| `plans/PR-Billing-Partial-Refund-Live-Adapter.md` | 90 |
| `tests/test_atlas_billing_content_ops_deflection_stripe_paid.py` | 100 |
| **Total** | **190** |

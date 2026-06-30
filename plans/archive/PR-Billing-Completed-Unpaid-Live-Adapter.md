# PR-Billing-Completed-Unpaid-Live-Adapter

## Why this slice exists

The billing real-adapters lane is replacing fake DB-pool webhook tests with
live asyncpg/Postgres state assertions one money-path boundary at a time. #1889
proved the async-payment-failed path leaves the report locked; the adjacent
`checkout.session.completed` path with `payment_status="unpaid"` still relies
on the hand-written `_Pool` fake and SQL-call assertions.

Root cause: the unpaid-completed test verifies a fake `billing_events` insert
and fake in-memory report/delivery dictionaries. That does not prove the real
webhook records the event without unlocking the report or queueing delivery.
This PR fixes that root for one more fail-closed payment boundary.

## Scope (this PR)

Ownership lane: real-adapters/test-quality
Slice phase: Production hardening

1. Convert
   `test_stripe_webhook_deflection_completed_unpaid_stays_pending` from
   `_Pool`/SQL-call assertions to the live asyncpg/Postgres harness.
2. Reuse the live locked-state helper from #1889 to assert real persisted state:
   report remains unpaid, no payment reference is written, no delivery row is
   queued, and the billing event is recorded once.
3. Ratchet `atlas_brain/api/billing.py` INTERNAL_MOCK only by the earned count
   from the removed fake-pool test.

### Review Contract

- The converted test uses the real asyncpg pool wrapper, applies the live
  billing migrations, seeds `saas_accounts` and a real deflection report row,
  and cleans up in `finally`.
- The test asserts real persisted effects: event recorded, report remains
  locked, delivery remains absent, and duplicate webhook handling is idempotent.
- Stripe remains mocked only at the external SDK/webhook-construction boundary.
- Remaining `_Pool` tests stay detector-visible for later slices.

### Files touched

- `plans/PR-Billing-Completed-Unpaid-Live-Adapter.md`
- `tests/maturity_sweep/baseline_atlas_brain_api.json`
- `tests/test_atlas_billing_content_ops_deflection_stripe_paid.py`

## Mechanism

Build the existing unpaid completed Checkout Session event with real
`account_id`, `request_id`, and `session_id` values. Seed a real report row via
`PostgresDeflectionReportArtifactStore`, run the actual `billing.stripe_webhook`
entrypoint through `_run_stripe_webhook`, and verify the persisted locked state
with `_assert_live_deflection_report_locked`. Then run the same webhook a second
time to prove idempotency against the real `billing_events` table.

## Intentional

- This is one fake-pool burn-down, not a broad webhook-test sweep.
- The existing live helper from #1889 is reused instead of adding another custom
  assertion shape.

## Deferred

- Remaining billing fake-pool webhook tests and the temporary
  `_resolve_billing_db_pool` direct-call shim are deferred to follow-up
  real-adapter burn-down slices.

Parked hardening: none.

## Verification

- Python compile check for `tests/test_atlas_billing_content_ops_deflection_stripe_paid.py` - passed.
- `ATLAS_MIGRATION_TEST_DATABASE_URL=postgresql://atlas:atlas_dev_password@localhost:5433/atlas python -m pytest tests/test_atlas_billing_content_ops_deflection_stripe_paid.py::test_stripe_webhook_deflection_completed_unpaid_stays_pending -q` - passed, 1 test.
- `python -m pytest tests/test_atlas_billing_content_ops_deflection_stripe_paid.py -q` - passed, 54 passed / 4 skipped.
- `ATLAS_MIGRATION_TEST_DATABASE_URL=postgresql://atlas:atlas_dev_password@localhost:5433/atlas python -m pytest tests/test_atlas_billing_content_ops_deflection_stripe_paid.py -q` - passed, 58 passed.
- `python scripts/maturity_sweep.py atlas_brain/api --tests-root tests --baseline tests/maturity_sweep/baseline_atlas_brain_api.json --top 80` - passed; `billing.py` is now `INTERNAL_MOCK x50`, score 226.

## Estimated diff size

| File | LOC |
|---|---:|
| `plans/PR-Billing-Completed-Unpaid-Live-Adapter.md` | 84 |
| `tests/maturity_sweep/baseline_atlas_brain_api.json` | 4 |
| `tests/test_atlas_billing_content_ops_deflection_stripe_paid.py` | 99 |
| **Total** | **187** |

# PR-Billing-Async-Failure-Live-Adapter

## Why this slice exists

The real-adapters/test-quality lane is burning down billing fake-pool debt only
when fake database behavior is replaced by real persisted-state verification.
#1887 and #1888 proved the paid checkout routes with live asyncpg/Postgres tests;
the adjacent async-payment-failed safety boundary still uses the hand-written
`_Pool` fake and SQL-call assertions.

Root cause: the failure-path test verifies that `_Pool.execute_calls` contains a
billing-event insert and that the fake in-memory report row remains unpaid. That
does not prove the real webhook keeps a failed async payment from unlocking a
report or queueing delivery. This PR fixes that root for one failure path and
adds small shared live helpers so the paid-state contract from #1887/#1888 does
not drift as more event types convert.

Diff-size note: this lands slightly over the 400 LOC soft cap because it pairs
one fake-pool conversion with the small helper extraction the #1888 review
flagged before adding a third copy of the same live paid-state assertions. The
work remains single-file test code plus the earned baseline update.

## Scope (this PR)

Ownership lane: real-adapters/test-quality
Slice phase: Production hardening

1. Convert
   `test_stripe_webhook_deflection_async_failure_is_observed_without_unlock`
   from `_Pool`/SQL-call assertions to the live asyncpg/Postgres harness.
2. Add focused live helper functions for seeding deflection report rows and
   asserting paid persisted state, then reuse them in the existing two live paid
   tests.
3. Ratchet `atlas_brain/api/billing.py` INTERNAL_MOCK only by the earned count
   from the removed fake-pool failure-path test.

### Review Contract

- The converted async-failure test uses the real asyncpg pool wrapper, applies
  the live billing migrations, seeds `saas_accounts` and a real deflection
  report row, and cleans up in `finally`.
- The failure test asserts real persisted effects: report remains unpaid,
  delivery is not queued, billing event is recorded once, and duplicate delivery
  is idempotent.
- The existing paid live tests share the same paid-state assertion helper rather
  than copy-pasting the row checks a third time.
- Stripe remains mocked only at the external SDK/webhook-construction boundary.
- Remaining `_Pool` tests stay detector-visible for later slices.

### Files touched

- `plans/PR-Billing-Async-Failure-Live-Adapter.md`
- `tests/maturity_sweep/baseline_atlas_brain_api.json`
- `tests/test_atlas_billing_content_ops_deflection_stripe_paid.py`

## Mechanism

Reuse the live billing harness added in #1887:

1. Add helper functions for live report seeding and paid-state assertions.
2. Refactor the two existing live paid webhook tests to use the paid-state
   helper without changing their event-specific assertions.
3. Convert the async-payment-failed route test to build the same Stripe event
   fixture, run the real `billing.stripe_webhook` entrypoint with a live pool,
   and assert the report stays locked while the billing event is recorded once.

## Intentional

- This is not a broad fake-pool cleanup. One more webhook route is converted so
  the maturity baseline drop remains reviewable.
- The helper extraction is limited to live billing fixtures/assertions already
  duplicated by #1887/#1888; production code is unchanged.

## Deferred

- Remaining billing fake-pool webhook tests and the temporary
  `_resolve_billing_db_pool` direct-call shim are deferred to follow-up
  real-adapter burn-down slices.

Parked hardening: none.

## Verification

- Python compile check for `tests/test_atlas_billing_content_ops_deflection_stripe_paid.py` - passed.
- `ATLAS_MIGRATION_TEST_DATABASE_URL=postgresql://atlas:atlas_dev_password@localhost:5433/atlas python -m pytest tests/test_atlas_billing_content_ops_deflection_stripe_paid.py::test_stripe_webhook_deflection_async_failure_is_observed_without_unlock -q` - passed, 1 test.
- `python -m pytest tests/test_atlas_billing_content_ops_deflection_stripe_paid.py -q` - passed, 55 passed / 3 skipped.
- `ATLAS_MIGRATION_TEST_DATABASE_URL=postgresql://atlas:atlas_dev_password@localhost:5433/atlas python -m pytest tests/test_atlas_billing_content_ops_deflection_stripe_paid.py -q` - passed, 58 passed.
- `python scripts/maturity_sweep.py atlas_brain/api --tests-root tests --baseline tests/maturity_sweep/baseline_atlas_brain_api.json --top 80` - passed; `billing.py` is now `INTERNAL_MOCK x53`, score 238.

## Estimated diff size

| File | LOC |
|---|---:|
| `plans/PR-Billing-Async-Failure-Live-Adapter.md` | 97 |
| `tests/maturity_sweep/baseline_atlas_brain_api.json` | 4 |
| `tests/test_atlas_billing_content_ops_deflection_stripe_paid.py` | 343 |
| **Total** | **444** |

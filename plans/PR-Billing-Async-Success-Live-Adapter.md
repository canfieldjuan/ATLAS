# PR-Billing-Async-Success-Live-Adapter

## Why this slice exists

The real-adapters/test-quality lane is burning down billing fake-pool debt only
when the fake is actually removed. #1887 established the template: keep Stripe
mocked at the external SDK/signature boundary, but run the application webhook
against a real asyncpg/Postgres adapter and assert persisted state.

Root cause: the async-payment-succeeded deflection webhook test still uses the
hand-written `_Pool` fake and verifies SQL call records (`execute_calls`) instead
of observable database state. That can pass while the real payment persistence
path is broken. This PR fixes that root for one thin webhook path rather than
relocating the fake into an uncounted injection shape.

## Scope (this PR)

Ownership lane: real-adapters/test-quality
Slice phase: Production hardening

1. Convert the
   `checkout.session.async_payment_succeeded` deflection webhook route test from
   `_Pool`/SQL-call assertions to the live asyncpg/Postgres harness introduced
   in #1887.
2. Ratchet `atlas_brain/api/billing.py` INTERNAL_MOCK only by the earned count
   from that removed fake-pool test.
3. Leave the remaining fake-pool tests detector-visible for later slices.

### Review Contract

- The converted test uses the real asyncpg pool wrapper, applies the live
  billing migrations, seeds `saas_accounts` and a real deflection report row,
  and cleans up in `finally`.
- The test asserts persisted effects: paid report row, queued delivery row,
  billing event insert, and idempotent re-delivery.
- Stripe remains mocked only at the external SDK/webhook-construction boundary.
- No consumer hand-editing or fake relocation counts as burn-down.

### Files touched

- `plans/PR-Billing-Async-Success-Live-Adapter.md`
- `tests/maturity_sweep/baseline_atlas_brain_api.json`
- `tests/test_atlas_billing_content_ops_deflection_stripe_paid.py`

## Mechanism

Reuse the live billing helpers added in #1887:

1. Build the existing async-success Stripe event fixture.
2. Connect to `ATLAS_MIGRATION_TEST_DATABASE_URL` through asyncpg, then apply
   the billing migrations required by the webhook path.
3. Seed `saas_accounts` plus a saved deflection report via
   `PostgresDeflectionReportArtifactStore`.
4. Call the real `billing.stripe_webhook` entrypoint through
   `_run_stripe_webhook`, with Stripe still faked at the SDK module boundary.
5. Assert the real tables show the paid report, pending delivery, one billing
   event, and idempotent duplicate handling.

## Intentional

- This does not attempt a broad test rewrite. The lane is intentionally
  ratcheted one fake-pool case at a time so every baseline drop is reviewable.
- The remaining `_Pool` tests stay as-is and stay counted by the maturity sweep
  until their own live-adapter slices land.

## Deferred

- Remaining billing fake-pool webhook tests and the temporary
  `_resolve_billing_db_pool` direct-call shim are deferred to follow-up
  real-adapter burn-down slices.

Parked hardening: none.

## Verification

- Python compile check for `tests/test_atlas_billing_content_ops_deflection_stripe_paid.py` - passed.
- `ATLAS_MIGRATION_TEST_DATABASE_URL=postgresql://atlas:atlas_dev_password@localhost:5433/atlas python -m pytest tests/test_atlas_billing_content_ops_deflection_stripe_paid.py::test_stripe_webhook_routes_deflection_async_success_to_paid_gate -q` - passed, 1 test.
- `python -m pytest tests/test_atlas_billing_content_ops_deflection_stripe_paid.py -q` - passed, 56 passed / 2 skipped.
- `ATLAS_MIGRATION_TEST_DATABASE_URL=postgresql://atlas:atlas_dev_password@localhost:5433/atlas python -m pytest tests/test_atlas_billing_content_ops_deflection_stripe_paid.py -q` - passed, 58 passed.
- `python scripts/maturity_sweep.py atlas_brain/api --tests-root tests --baseline tests/maturity_sweep/baseline_atlas_brain_api.json --top 80` - passed; `billing.py` is now `INTERNAL_MOCK x56`, score 250.

## Estimated diff size

| File | LOC |
|---|---:|
| `plans/PR-Billing-Async-Success-Live-Adapter.md` | 89 |
| `tests/maturity_sweep/baseline_atlas_brain_api.json` | 4 |
| `tests/test_atlas_billing_content_ops_deflection_stripe_paid.py` | 143 |
| **Total** | **236** |

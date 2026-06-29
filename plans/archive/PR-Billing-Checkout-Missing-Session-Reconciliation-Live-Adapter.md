# PR-Billing-Checkout-Missing-Session-Reconciliation-Live-Adapter

## Why this slice exists

The real-adapters/test-quality lane is burning down billing tests that still
prove money-path behavior with hand-written DB-pool fakes. #1905 converted the
aged paid-but-missing Checkout reconciliation test and explicitly deferred the
missing-session variant once migrations 336/337 were enrolled in the live test
harness.

Root cause: `test_deflection_reconciliation_binds_empty_string_for_missing_session_id`
asserts the behavior by sniffing `_Pool.execute_calls` and checking bound SQL
arguments. That proves the fake saw `""`, but it does not prove Postgres
enforces the real invariant that `stripe_session_id` is stored as non-NULL
`""` and dedupes on `(account_id, request_id, stripe_session_id)` during a
Stripe retry.

This change fixes the root for this edge by moving the test to the same live
asyncpg/Postgres adapter harness as the adjacent reconciliation test and
asserting persisted ledger state plus retry idempotency.

## Scope (this PR)

Ownership lane: real-adapters/test-quality
Slice phase: Production hardening

1. Convert only `test_deflection_reconciliation_binds_empty_string_for_missing_session_id`
   from `_Pool` SQL-call assertions to a live Postgres reconciliation ledger
   assertion.
2. Prove the aged missing-report Checkout handler records `stripe_session_id`
   as `""` for an empty session id, never `NULL`.
3. Prove retry idempotency by invoking the handler twice and asserting one
   reconciliation row remains.
4. Preserve the no-delivery side effect for the account.

### Review Contract

- Acceptance criteria:
  - The converted test uses `_connect_live_billing_pool()` and
    `_apply_live_billing_migrations()`, not `_Pool`.
  - The test asserts the persisted reconciliation row shape, including
    `stripe_session_id == ""`.
  - The test asserts `COUNT(*) == 1` after the first handler call and again
    after a same-event retry.
  - The test asserts no report delivery was queued for the account.
- Affected surfaces:
  - `tests/test_atlas_billing_content_ops_deflection_stripe_paid.py`
- Risk areas:
  - Billing/payment reconciliation idempotency.
  - Empty Stripe session id handling in the unique ledger key.
  - Live test cleanup isolation.
- Reviewer rules:
  - R1 Requirements match.
  - R2 Test evidence.
  - R4 Data integrity / persistence.
  - R8 Error / edge cases.
  - R11 Money path.
  - R14 Codebase verification.

### Files touched

- `plans/PR-Billing-Checkout-Missing-Session-Reconciliation-Live-Adapter.md`
- `tests/test_atlas_billing_content_ops_deflection_stripe_paid.py`

## Mechanism

The test creates a synthetic Checkout session with `session_id=""`, applies the
live billing migrations, cleans account/request rows, and calls
`_handle_content_ops_deflection_report_checkout_completed()` with an aged event
so the handler takes the permanent paid-but-missing reconciliation path.

Instead of inspecting fake SQL calls, the test reads
`content_ops_deflection_paid_reconciliation` through asyncpg and asserts the
row persisted with:

```python
{
    "account_id": account_id,
    "request_id": "req-live-checkout-missing-session-reconciliation",
    "stripe_session_id": "",
    "event_type": "checkout.session.completed",
    "reason": "paid_report_missing",
}
```

It then replays the same handler call and asserts the ledger count remains one,
which proves the real `ON CONFLICT`/migration 337 empty-string behavior rather
than the fake call path.

## Intentional

- This keeps direct handler coverage as the exercised boundary; full webhook
  event dedupe/audit coverage remains owned by the existing webhook tests.
- This does not modify production billing code. Migrations 336/337 and the
  `session_id or ""` path already landed; this slice replaces the fake proof
  with persisted-state proof.
- The Stripe Checkout session remains synthetic while the database adapter and
  reconciliation ledger state are real.

## Deferred

- Remaining billing fake-pool checkout/refund/dispute/delta tests and the
  temporary `_resolve_billing_db_pool` direct-call shim stay for follow-up
  slices.

Parked hardening: none.

## Verification

- `python -m py_compile tests/test_atlas_billing_content_ops_deflection_stripe_paid.py`
  - Pass.
- `ATLAS_MIGRATION_TEST_DATABASE_URL=postgresql://atlas:atlas@localhost:5433/atlas pytest tests/test_atlas_billing_content_ops_deflection_stripe_paid.py -k 'missing_session_id'`
  - Pass: 1 passed, 58 deselected.
- `ATLAS_MIGRATION_TEST_DATABASE_URL=postgresql://atlas:atlas@localhost:5433/atlas pytest tests/test_atlas_billing_content_ops_deflection_stripe_paid.py`
  - Pass: 59 passed.
- `python scripts/maturity_sweep.py atlas_brain/api --tests-root tests --top 10`
  - Pass/advisory: `billing.py` remains score 178 with `INTERNAL_MOCK x38`.

## Estimated diff size

| File | LOC |
|---|---:|
| `plans/PR-Billing-Checkout-Missing-Session-Reconciliation-Live-Adapter.md` | 125 |
| `tests/test_atlas_billing_content_ops_deflection_stripe_paid.py` | 104 |
| **Total** | **229** |

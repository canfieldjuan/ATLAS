# PR-Billing-Checkout-Race-Window-Live-Adapter

## Why this slice exists

The real-adapters/test-quality lane is burning down billing tests that still
prove money-path behavior with hand-written DB-pool fakes. #1905 and #1906
converted the aged paid-but-missing reconciliation cases to live Postgres
ledger assertions. The adjacent race-window test is the opposite invariant:
a recent missing-report Checkout event must raise 409 so Stripe retries, and it
must not write the permanent reconciliation ledger yet.

Root cause: `test_deflection_checkout_completion_retries_409_within_race_window`
asserts this money-path behavior by using `_Pool.update_result = "UPDATE 0"`
and checking the fake's `execute_calls` list for no reconciliation SQL. That
proves the fake did not see a string, but it does not prove the real asyncpg
adapter leaves `content_ops_deflection_paid_reconciliation` and deliveries
empty when the handler raises 409.

This change fixes the root for this edge by moving the test to the same live
asyncpg/Postgres adapter harness as the surrounding Checkout reconciliation
tests and asserting persisted absence of reconciliation and delivery rows.

## Scope (this PR)

Ownership lane: real-adapters/test-quality
Slice phase: Production hardening

1. Convert only `test_deflection_checkout_completion_retries_409_within_race_window`
   from `_Pool` SQL-call assertions to a live Postgres state assertion.
2. Prove a recent paid-but-missing Checkout event raises HTTP 409.
3. Prove the race-window path leaves the reconciliation ledger empty for that
   account/request/session.
4. Prove no report delivery was queued for the account.

### Review Contract

- Acceptance criteria:
  - The converted test uses `_connect_live_billing_pool()` and
    `_apply_live_billing_migrations()`, not `_Pool`.
  - The test asserts HTTP 409 for a recent event.
  - The test asserts zero `content_ops_deflection_paid_reconciliation` rows for
    the account/request.
  - The test asserts zero `content_ops_deflection_report_deliveries` rows for
    the account.
- Affected surfaces:
  - `tests/test_atlas_billing_content_ops_deflection_stripe_paid.py`
- Risk areas:
  - Billing/payment retry semantics.
  - Race-window distinction between transient write ordering and permanent
    paid-but-missing reconciliation.
  - Live test cleanup isolation.
- Reviewer rules:
  - R1 Requirements match.
  - R2 Test evidence.
  - R4 Data integrity / persistence.
  - R8 Error / edge cases.
  - R11 Money path.
  - R14 Codebase verification.

### Files touched

- `plans/PR-Billing-Checkout-Race-Window-Live-Adapter.md`
- `tests/test_atlas_billing_content_ops_deflection_stripe_paid.py`

## Mechanism

The test creates a synthetic Checkout session for a unique account/request,
applies the live billing migrations, cleans account/request rows, and calls
`_handle_content_ops_deflection_report_checkout_completed()` with
`event_created=int(time.time())`, keeping the event inside the configured
reconcile grace window.

The handler should raise HTTP 409. After the exception, the test reads live
Postgres state and asserts:

```python
SELECT COUNT(*) FROM content_ops_deflection_paid_reconciliation
WHERE account_id = $1 AND request_id = $2
```

returns `0`, and that the account has no delivery rows. This proves the real
adapter did not accidentally treat a transient race as a permanent
paid-but-missing case.

## Intentional

- This keeps direct handler coverage as the exercised boundary; full webhook
  event dedupe/audit coverage remains owned by the existing webhook tests.
- This does not modify production billing code. The slice replaces fake proof
  with persisted-state proof for existing race-window behavior.
- The Stripe Checkout session remains synthetic while the database adapter and
  reconciliation/delivery state are real.

## Deferred

- Remaining billing fake-pool checkout/refund/dispute/delta tests and the
  temporary `_resolve_billing_db_pool` direct-call shim stay for follow-up
  slices.

Parked hardening: none.

## Verification

- `python -m py_compile tests/test_atlas_billing_content_ops_deflection_stripe_paid.py`
  - Pass.
- `ATLAS_MIGRATION_TEST_DATABASE_URL=postgresql://atlas:atlas@localhost:5433/atlas pytest tests/test_atlas_billing_content_ops_deflection_stripe_paid.py -k 'race_window'`
  - Pass: 1 passed, 58 deselected.
- `ATLAS_MIGRATION_TEST_DATABASE_URL=postgresql://atlas:atlas@localhost:5433/atlas pytest tests/test_atlas_billing_content_ops_deflection_stripe_paid.py`
  - Pass: 59 passed.
- `python scripts/maturity_sweep.py atlas_brain/api --tests-root tests --top 10`
  - Pass/advisory: `billing.py` remains score 178 with `INTERNAL_MOCK x38`.

## Estimated diff size

| File | LOC |
|---|---:|
| `plans/PR-Billing-Checkout-Race-Window-Live-Adapter.md` | 116 |
| `tests/test_atlas_billing_content_ops_deflection_stripe_paid.py` | 65 |
| **Total** | **181** |

# PR-Billing-Checkout-Lower-Authorized-Amount-Live-Adapter

## Why this slice exists

The real-adapters/test-quality lane is burning down billing tests that still
prove money-path behavior with hand-written DB-pool fakes. The Checkout
completion reconciliation branch is now live-covered on both sides of the
grace window; the next remaining Checkout fake is the authorized lower amount
case.

Root cause: `test_deflection_checkout_completion_accepts_lower_authorized_amount`
uses `_Pool.add_report()` and checks `pool.execute_calls[0]` SQL arguments to
infer that the handler would accept a previously authorized partner/lower-price
Checkout amount. That verifies the fake's argument tuple, but it does not prove
the real `PostgresDeflectionReportArtifactStore.record_checkout_authorization()`
metadata and `mark_paid()` predicate work together or that the paid report and
delivery queue are persisted.

This change fixes the root for this edge by moving the test to the live
asyncpg/Postgres adapter harness and asserting observable report and delivery
state after the lower authorized amount is accepted.

## Scope (this PR)

Ownership lane: real-adapters/test-quality
Slice phase: Production hardening

1. Convert only `test_deflection_checkout_completion_accepts_lower_authorized_amount`
   from `_Pool` SQL argument assertions to a live Postgres state assertion.
2. Seed a real report and record real checkout authorization metadata for a
   lower partner amount.
3. Prove the Checkout handler marks that report paid with the Stripe session id.
4. Prove the delivery queue row is created for the paid report.

### Review Contract

- Acceptance criteria:
  - The converted test uses `_connect_live_billing_pool()` and
    `_apply_live_billing_migrations()`, not `_Pool`.
  - The test seeds a report via `_seed_live_deflection_report()` and records
    checkout authorization via `PostgresDeflectionReportArtifactStore`.
  - The test asserts persisted `paid = true` and the expected
    `payment_reference`.
  - The test asserts the delivery queue row exists with `delivery_status =
    "pending"` and the expected `payment_reference`.
- Affected surfaces:
  - `tests/test_atlas_billing_content_ops_deflection_stripe_paid.py`
- Risk areas:
  - Billing/payment authorization terms.
  - Lower authorized amount acceptance.
  - Report delivery queueing after accepted payment.
  - Live test cleanup isolation.
- Reviewer rules:
  - R1 Requirements match.
  - R2 Test evidence.
  - R4 Data integrity / persistence.
  - R8 Error / edge cases.
  - R11 Money path.
  - R14 Codebase verification.

### Files touched

- `plans/PR-Billing-Checkout-Lower-Authorized-Amount-Live-Adapter.md`
- `tests/test_atlas_billing_content_ops_deflection_stripe_paid.py`

## Mechanism

The test creates a unique account/request, applies live billing migrations,
seeds a report, and records checkout authorization with:

```python
price_variant="partner"
amount_cents=120000
currency="usd"
price_id="price_partner"
```

It temporarily configures the allowed Checkout amount set to include the lower
partner amount, invokes `_handle_content_ops_deflection_report_checkout_completed()`
with a synthetic paid Checkout session at `120000`, then reads live Postgres
state to assert the report is paid and a pending delivery row exists.

## Intentional

- This keeps direct handler coverage as the exercised boundary; full webhook
  event dedupe/audit coverage remains owned by the existing webhook tests.
- This does not modify production billing code. The slice replaces fake SQL
  argument proof with persisted-state proof for existing authorization behavior.
- The Stripe Checkout session remains synthetic while checkout authorization,
  report payment, and delivery queue state are real.

## Deferred

- Remaining billing fake-pool checkout/refund/dispute/delta tests and the
  temporary `_resolve_billing_db_pool` direct-call shim stay for follow-up
  slices.

Parked hardening: none.

## Verification

- `python -m py_compile tests/test_atlas_billing_content_ops_deflection_stripe_paid.py`
  - Pass.
- `ATLAS_MIGRATION_TEST_DATABASE_URL=postgresql://atlas:atlas@localhost:5433/atlas pytest tests/test_atlas_billing_content_ops_deflection_stripe_paid.py -k 'lower_authorized_amount'`
  - Pass: 1 passed, 58 deselected.
- `ATLAS_MIGRATION_TEST_DATABASE_URL=postgresql://atlas:atlas@localhost:5433/atlas pytest tests/test_atlas_billing_content_ops_deflection_stripe_paid.py`
  - Pass: 59 passed.
- `python scripts/maturity_sweep.py atlas_brain/api --tests-root tests --top 10`
  - Pass/advisory: `billing.py` remains score 178 with `INTERNAL_MOCK x38`.

## Estimated diff size

| File | LOC |
|---|---:|
| `plans/PR-Billing-Checkout-Lower-Authorized-Amount-Live-Adapter.md` | 114 |
| `tests/test_atlas_billing_content_ops_deflection_stripe_paid.py` | 93 |
| **Total** | **207** |

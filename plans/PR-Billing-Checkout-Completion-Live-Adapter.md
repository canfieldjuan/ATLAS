# PR-Billing-Checkout-Completion-Live-Adapter

## Why this slice exists

The billing real-adapters lane is replacing hand-written fake DB-pool webhook
and handler tests with live asyncpg/Postgres state assertions one money-path
boundary at a time. The checkout-completion happy path still proves "mark the
deflection report paid and queue delivery" with `_Pool.execute_calls` and fake
`delivery_rows`.

Root cause: `test_deflection_checkout_completion_marks_report_paid` asserts SQL
strings and fake call arguments instead of proving the real asyncpg/Postgres
adapter updates `content_ops_deflection_reports` and upserts
`content_ops_deflection_report_deliveries`. This PR fixes the root for that one
handler path by running `_handle_content_ops_deflection_report_checkout_completed`
against a migrated Postgres database and asserting persisted report/delivery
state.

## Scope (this PR)

Ownership lane: real-adapters/test-quality
Slice phase: Production hardening

1. Convert `test_deflection_checkout_completion_marks_report_paid` from
   `_Pool`/SQL-call assertions to the live asyncpg/Postgres harness.
2. Preserve the current behavior contract: the handler returns `None`, marks the
   report paid with the Stripe session reference, and creates one pending
   delivery row without exposing the delivery email in the row body.
3. Keep maturity-sweep honest: only ratchet `atlas_brain/api/billing.py` if this
   conversion earns a detector-visible reduction. Remaining `_Pool` tests stay
   grep-visible for later burn-down slices.

### Review Contract

Acceptance criteria:

- The converted test uses the real asyncpg pool wrapper, applies live billing
  migrations, seeds `saas_accounts` plus an unpaid deflection report row, and
  cleans up in `finally`.
- The test calls the real `_handle_content_ops_deflection_report_checkout_completed`
  handler, not the full webhook, because this slice is replacing the direct
  handler fake-pool proof.
- Report and delivery state are asserted from Postgres; no SQL-string or fake
  call-list assertions remain in this test.
- The test asserts the report is paid with the expected `payment_reference`,
  delivery status is `pending`, and the delivery row does not store the buyer
  email.
- Remaining `_Pool` tests stay detector-visible for later slices.

Affected surfaces:

- `tests/test_atlas_billing_content_ops_deflection_stripe_paid.py` only.
- Runtime billing code is exercised through the existing direct handler, but
  production implementation files are intentionally unchanged.

Risk areas:

- Billing checkout completion on the paid report unlock path.
- Delivery upsert state must remain idempotent and not persist buyer email in
  the delivery queue row.
- The live adapter proof must not weaken the old fake-pool happy-path contract.

Reviewer rules: R1, R2, R3, R6, R8, R10, R13, R14.

### Files touched

- `plans/PR-Billing-Checkout-Completion-Live-Adapter.md`
- `tests/test_atlas_billing_content_ops_deflection_stripe_paid.py`

## Mechanism

Seed a live unpaid report through `PostgresDeflectionReportArtifactStore`, then
call `_handle_content_ops_deflection_report_checkout_completed` with the existing
synthetic Stripe Checkout session. Query `content_ops_deflection_reports` and
`content_ops_deflection_report_deliveries` to prove the handler marked the
report paid, set `payment_reference`, created one pending delivery row, and did
not store `buyer@example.com` in the delivery row.

## Intentional

- This is one direct-handler fake-pool burn-down, not the whole checkout helper
  cluster.
- This slice does not insert or assert `billing_events`; the direct handler does
  not own event dedupe/audit logging.
- The Stripe session remains synthetic; the database adapter and persisted state
  are real.

## Deferred

- Remaining billing fake-pool checkout/refund/dispute/delta tests and the
  temporary `_resolve_billing_db_pool` direct-call shim are deferred to follow-up
  real-adapter burn-down slices.

Parked hardening: none.

## Verification

- Python compile check for
  `tests/test_atlas_billing_content_ops_deflection_stripe_paid.py` with
  `python -m py_compile` - passed.
- `ATLAS_MIGRATION_TEST_DATABASE_URL=postgresql://atlas:atlas_dev_password@localhost:5433/atlas python -m pytest tests/test_atlas_billing_content_ops_deflection_stripe_paid.py::test_deflection_checkout_completion_marks_report_paid -q` - passed, 1 test.
- `python -m pytest tests/test_atlas_billing_content_ops_deflection_stripe_paid.py -q` - passed, 42 passed / 16 skipped.
- `ATLAS_MIGRATION_TEST_DATABASE_URL=postgresql://atlas:atlas_dev_password@localhost:5433/atlas python -m pytest tests/test_atlas_billing_content_ops_deflection_stripe_paid.py -q` - passed, 58 passed.
- `python scripts/maturity_sweep.py atlas_brain/api --tests-root tests --baseline tests/maturity_sweep/baseline_atlas_brain_api.json --top 80` - passed; no baseline update because `billing.py` remained `INTERNAL_MOCK x38`, score 178.

## Estimated diff size

| File | LOC |
|---|---:|
| `plans/PR-Billing-Checkout-Completion-Live-Adapter.md` | 112 |
| `tests/test_atlas_billing_content_ops_deflection_stripe_paid.py` | 80 |
| **Total** | **192** |

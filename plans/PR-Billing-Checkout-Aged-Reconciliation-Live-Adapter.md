# PR-Billing-Checkout-Aged-Reconciliation-Live-Adapter

## Why this slice exists

The billing real-adapters lane is replacing hand-written fake DB-pool Stripe
money-path tests with live asyncpg/Postgres state assertions. The previous
missing-report slice converted the fresh/unknown-age 409 retry path and
explicitly deferred the aged paid-but-missing reconciliation branch because that
path needs the reconciliation ledger migrations.

Root cause: `test_deflection_checkout_completion_records_reconciliation_when_event_aged`
asserts fake `execute_calls` SQL strings instead of proving the real
asyncpg/Postgres adapter writes `content_ops_deflection_paid_reconciliation` and
returns 2xx for an aged paid-but-missing Checkout event. This PR fixes the root
for that one handler path by applying migrations 336/337 in the live harness and
asserting the persisted reconciliation ledger row.

## Scope (this PR)

Ownership lane: real-adapters/test-quality
Slice phase: Production hardening

1. Add the paid reconciliation ledger migrations to the live billing test
   migration set.
2. Convert `test_deflection_checkout_completion_records_reconciliation_when_event_aged`
   from `_Pool`/SQL-call assertions to the live asyncpg/Postgres harness.
3. Preserve the behavior contract: an aged paid-but-missing Checkout event
   returns `None`/2xx, records one reconciliation ledger row, and queues no
   delivery row.
4. Enroll the reconciliation migrations in the Stripe paid workflow path
   filters, so migration-only PRs rerun this live billing suite.
5. Keep maturity-sweep honest: only ratchet `atlas_brain/api/billing.py` if this
   conversion earns a detector-visible reduction. Remaining `_Pool` tests stay
   grep-visible for later burn-down slices.

### Review Contract

Acceptance criteria:

- The live migration set includes
  `336_content_ops_deflection_paid_reconciliation.sql` and
  `337_content_ops_deflection_reconciliation_null_session.sql`.
- The converted test uses the real asyncpg pool wrapper, applies live billing
  migrations, cleans any existing rows for the generated account/request, and
  closes the pool in `finally`.
- The test calls the real `_handle_content_ops_deflection_report_checkout_completed`
  handler with an event timestamp older than the configured grace window.
- No deflection report row is seeded for the account/request under test.
- The handler returns `None` instead of raising 409.
- State is asserted from Postgres: exactly one reconciliation row exists with
  account id, request id, Stripe session id, event type, and
  `paid_report_missing`; no delivery row exists for the account.
- The test asserts `COUNT(*) == 1` after the first ledger write, then reruns the
  same aged event and asserts the count remains `1` to prove retry-storm
  idempotency.
- The incident log still records `paid_report_missing_after_payment` with
  `disposition="reconciled"`.
- The workflow path filters include migrations 336/337 under both
  `pull_request` and `push`.
- Remaining `_Pool` tests stay detector-visible for later slices.

Affected surfaces:

- `.github/workflows/atlas_content_ops_deflection_stripe_paid_checks.yml`
- `tests/test_atlas_billing_content_ops_deflection_stripe_paid.py`
- Runtime billing code is exercised through the existing direct handler, but
  production implementation files are intentionally unchanged.

Risk areas:

- Billing checkout completion when Stripe confirms payment and the report row
  still has not appeared after the write-ordering grace window.
- The aged path must stop Stripe retry storms by returning 2xx while still
  recording a durable manual-reconciliation row.
- CI enrollment must follow the new live migration dependencies, or migration
  regressions can bypass the Stripe paid live suite.
- The live adapter proof must not weaken the old fake-pool reconciliation
  contract.

Reviewer rules: R1, R2, R3, R6, R8, R10, R13, R14.

### Files touched

- `.github/workflows/atlas_content_ops_deflection_stripe_paid_checks.yml`
- `plans/PR-Billing-Checkout-Aged-Reconciliation-Live-Adapter.md`
- `tests/test_atlas_billing_content_ops_deflection_stripe_paid.py`

## Mechanism

Extend `_LIVE_BILLING_MIGRATIONS` with the reconciliation ledger migrations so
the live test database has the same table and NULL-safe session-id constraint as
runtime. Clean the generated account/request and intentionally do not seed a
report row. Call `_handle_content_ops_deflection_report_checkout_completed` with
an aged `event_created` timestamp, then query
`content_ops_deflection_paid_reconciliation` and
`content_ops_deflection_report_deliveries` to prove the handler recorded the
manual reconciliation case and did not queue delivery. Query `COUNT(*)` for the
ledger row after the first call, then rerun the same aged event and assert the
count remains `1`, proving the `ON CONFLICT` idempotency behavior without
SQL-string assertions. Add migrations 336/337 to both workflow path-filter lists
so changes to the ledger schema rerun this suite.

## Intentional

- This is one direct-handler fake-pool burn-down, not the whole missing-report
  reconciliation cluster.
- The empty-string missing-session dedup test remains deferred to the next slice;
  this PR introduces the live ledger migrations and proves the normal session-id
  aged path first.
- This slice does not insert or assert `billing_events`; the direct handler does
  not own event dedupe/audit logging.
- The Stripe session remains synthetic; the database adapter and persisted state
  are real.

## Deferred

- Live conversion of `test_deflection_reconciliation_binds_empty_string_for_missing_session_id`
  is deferred to a follow-up slice now that the live ledger migrations are in
  the harness.
- Remaining billing fake-pool checkout/refund/dispute/delta tests and the
  temporary `_resolve_billing_db_pool` direct-call shim are deferred to follow-up
  real-adapter burn-down slices.

Parked hardening: none.

## Verification

- Python compile check for
  `tests/test_atlas_billing_content_ops_deflection_stripe_paid.py` with
  `python -m py_compile` - passed.
- `ATLAS_MIGRATION_TEST_DATABASE_URL=postgresql://atlas:atlas_dev_password@localhost:5433/atlas python -m pytest tests/test_atlas_billing_content_ops_deflection_stripe_paid.py::test_deflection_checkout_completion_records_reconciliation_when_event_aged -q` - passed, 1 test.
- `python -m pytest tests/test_atlas_billing_content_ops_deflection_stripe_paid.py -q` - passed, 39 passed / 20 skipped.
- `ATLAS_MIGRATION_TEST_DATABASE_URL=postgresql://atlas:atlas_dev_password@localhost:5433/atlas python -m pytest tests/test_atlas_billing_content_ops_deflection_stripe_paid.py -q` - passed, 59 passed.
- `python scripts/maturity_sweep.py atlas_brain/api --tests-root tests --baseline tests/maturity_sweep/baseline_atlas_brain_api.json --top 80` - passed; no baseline update because `billing.py` remained `INTERNAL_MOCK x38`, score 178.

## Estimated diff size

| File | LOC |
|---|---:|
| `.github/workflows/atlas_content_ops_deflection_stripe_paid_checks.yml` | 4 |
| `plans/PR-Billing-Checkout-Aged-Reconciliation-Live-Adapter.md` | 143 |
| `tests/test_atlas_billing_content_ops_deflection_stripe_paid.py` | 127 |
| **Total** | **274** |

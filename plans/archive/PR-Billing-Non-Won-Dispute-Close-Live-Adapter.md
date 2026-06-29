# PR-Billing-Non-Won-Dispute-Close-Live-Adapter

## Why this slice exists

The billing real-adapters lane is replacing hand-written fake DB-pool webhook
tests with live asyncpg/Postgres state assertions one money-path boundary at a
time. #1893 covered dispute-created relock, and the non-won dispute-close path
still proves "observe but do not restore" with `_Pool` dictionaries and
SQL-call filtering.

Root cause: `test_stripe_webhook_non_won_dispute_close_does_not_restore_report`
asserts fake `report_rows`, delivery dictionaries, and `execute_calls` instead
of real persisted report, delivery, and billing-event state. That does not
prove the real webhook leaves an unpaid/revoked report untouched for a lost
dispute while still auditing the closed dispute event. This PR fixes the root
for that one no-restore path by running the real webhook against a migrated
Postgres database.

## Scope (this PR)

Ownership lane: real-adapters/test-quality
Slice phase: Production hardening

1. Convert
   `test_stripe_webhook_non_won_dispute_close_does_not_restore_report` from
   `_Pool`/SQL-call assertions to the live asyncpg/Postgres harness.
2. Preserve the current behavior contract: non-won dispute close does not call
   Stripe Checkout Session lookup, does not restore the report, does not create
   delivery rows, records the billing event once, and logs that the dispute was
   closed without restore.
3. Keep maturity-sweep honest: do not ratchet
   `atlas_brain/api/billing.py` unless the detector reports an earned
   reduction. Remaining `_Pool` tests stay grep-visible for later burn-down
   slices.

### Review Contract

Acceptance criteria:

- The converted test uses the real asyncpg pool wrapper, applies live billing
  migrations, seeds `saas_accounts` and an unpaid deflection report row with a
  preserved payment reference, and cleans up in `finally`.
- Stripe remains mocked only at the external webhook/checkout-session SDK
  boundary; the test still asserts no checkout-session lookup happens for a
  non-won dispute.
- The test asserts persisted report state (`paid=false`, `paid_at=NULL`,
  payment reference preserved), account-wide absence of delivery rows, and
  exactly one `billing_events` row for the observed dispute close.
- The test proves the non-won dispute close is a report-row no-op by asserting
  `updated_at` is unchanged across webhook handling.
- Remaining `_Pool` tests stay detector-visible for later slices.

Affected surfaces:

- `tests/test_atlas_billing_content_ops_deflection_stripe_paid.py` only.
- Runtime Stripe webhook code is exercised through the existing real endpoint
  harness, but production implementation files are intentionally unchanged.

Risk areas:

- Billing and webhook idempotency on a money-path event.
- Revoked paid-report access must not be restored for lost or non-won disputes.
- Live adapter coverage must not weaken the old fake-pool no-write/no-delivery
  contract while removing SQL-call assertions.

Reviewer rules: R1, R2, R3, R6, R8, R10, R13, R14.

### Files touched

- `plans/PR-Billing-Non-Won-Dispute-Close-Live-Adapter.md`
- `tests/test_atlas_billing_content_ops_deflection_stripe_paid.py`

## Mechanism

Seed a live report through `PostgresDeflectionReportArtifactStore`, set it to a
paid-then-unpaid state with a stable payment reference to match the pre-existing
revoked access shape, and build the existing `charge.dispute.closed` event with
`status="lost"`. Run `_run_stripe_webhook` against the live pool and query
Postgres to prove the report remains unpaid, no delivery rows exist for the
account, and the webhook event was still inserted into `billing_events`.

## Intentional

- This is one fake-pool burn-down, not the whole dispute lifecycle cluster.
- The checkout-session list remains a fake Stripe SDK boundary because the
  behavior under test is that non-won disputes return before any lookup.

## Deferred

- Remaining billing fake-pool refund/dispute tests and the temporary
  `_resolve_billing_db_pool` direct-call shim are deferred to follow-up
  real-adapter burn-down slices.

Parked hardening: none.

## Verification

- Python compile check for `tests/test_atlas_billing_content_ops_deflection_stripe_paid.py` with `python -m py_compile` - passed.
- `ATLAS_MIGRATION_TEST_DATABASE_URL=postgresql://atlas:atlas_dev_password@localhost:5433/atlas python -m pytest tests/test_atlas_billing_content_ops_deflection_stripe_paid.py::test_stripe_webhook_non_won_dispute_close_does_not_restore_report -q` - passed, 1 test.
- `python -m pytest tests/test_atlas_billing_content_ops_deflection_stripe_paid.py -q` - passed, 48 passed / 10 skipped.
- `ATLAS_MIGRATION_TEST_DATABASE_URL=postgresql://atlas:atlas_dev_password@localhost:5433/atlas python -m pytest tests/test_atlas_billing_content_ops_deflection_stripe_paid.py -q` - passed, 58 passed.
- `python scripts/maturity_sweep.py atlas_brain/api --tests-root tests --baseline tests/maturity_sweep/baseline_atlas_brain_api.json --top 80` - passed; `billing.py` remains `INTERNAL_MOCK x47`, score 214, so no baseline change was earned.

## Estimated diff size

| File | LOC |
|---|---:|
| `plans/PR-Billing-Non-Won-Dispute-Close-Live-Adapter.md` | 110 |
| `tests/test_atlas_billing_content_ops_deflection_stripe_paid.py` | 124 |
| **Total** | **234** |

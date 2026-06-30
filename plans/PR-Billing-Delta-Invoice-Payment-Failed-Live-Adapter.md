# PR-Billing-Delta-Invoice-Payment-Failed-Live-Adapter

## Why this slice exists

Issue #1877 is burning down tests that mock first-party billing persistence
instead of exercising the real DB adapter. #1910 converted the delta
`customer.subscription.updated` grant path. The next fake-pool delta webhook is
`invoice.payment_failed`, which proves entitlement revocation but still uses the
hand-written `_Pool` fake for billing-event persistence and account mutation
guards.

Root cause: `test_stripe_webhook_delta_invoice_payment_failed_revokes_entitlement`
uses `_Pool.execute_calls` as the persistence proof and as the no-account-update
guard. That proves a fake SQL recorder, not the real asyncpg/Postgres adapter.

This PR fixes the root for one webhook event (`invoice.payment_failed`) by
replacing `_Pool` with the live billing DB adapter, persisted `billing_events`
assertions, and a live before/after `saas_accounts` no-mutation assertion.

## Scope (this PR)

Ownership lane: real-adapters/test-quality
Slice phase: Production hardening

1. Convert
   `test_stripe_webhook_delta_invoice_payment_failed_revokes_entitlement` from
   `_Pool` to `_connect_live_billing_pool()`.
2. Assert the webhook still revokes deflection-delta entitlement through the
   in-memory artifact store while the billing event is persisted in the live
   database with the expected account, event id, event type, and payload.
3. Preserve the prior no-`saas_accounts`-mutation invariant as a live
   before/after row-state assertion.
4. Keep Stripe itself mocked at the webhook boundary; this is a billing
   persistence/adapter proof, not a Stripe API integration test.

### Review Contract

- Acceptance criteria:
  - The converted test has `@pytest.mark.integration`.
  - The converted test no longer instantiates `_Pool` or asserts on
    `execute_calls`.
  - The entitlement revocation remains observable through
    `list_deflection_delta_entitled_account_ids()`.
  - The test proves `saas_accounts` plan/status/customer/subscription fields do
    not change during a delta invoice payment-failed webhook.
  - The billing event is asserted from Postgres, not from fake SQL call capture.
- Affected surfaces:
  - Stripe webhook tests for deflection-delta invoice entitlement revocation.
  - Live billing test helpers in
    `tests/test_atlas_billing_content_ops_deflection_stripe_paid.py`.
- Risk areas:
  - Accidentally widening cleanup beyond the test account/event.
  - Reintroducing constant values for unique Stripe customer/subscription
    sentinel fields.
  - Leaving the live DB test in the non-integration unit backstop.
- Triggered reviewer rules:
  - R2 Test evidence.
  - R3 Security/auth/payment path.
  - R8 Stateful persistence.
  - R14 Codebase verification.

### Files touched

- `plans/PR-Billing-Delta-Invoice-Payment-Failed-Live-Adapter.md`
- `tests/test_atlas_billing_content_ops_deflection_stripe_paid.py`

## Mechanism

- In the converted test:
  - connect to the live migration test DB;
  - apply the billing migrations;
  - delete prior rows for the generated account/event with the existing
    account-scoped live cleanup helper;
  - seed a `saas_accounts` row with account-scoped sentinel
    plan/status/customer/subscription values for the `billing_events` account
    FK and no-mutation proof;
  - seed an active in-memory delta entitlement;
  - run the webhook with the same mocked Stripe module and in-memory entitlement
    store;
  - assert the entitlement store no longer returns the account;
  - assert the seeded `saas_accounts` fields are unchanged after the webhook;
  - assert `billing_events` contains exactly one row for the delta invoice
    payment-failed event, account, event type, and JSON payload.
- Close the live pool in a `finally` block.

## Intentional

- The entitlement artifact store remains in-memory because this slice is focused
  on replacing billing DB persistence. Later slices can decide whether to
  exercise the Postgres artifact store for entitlement rows.
- This does not delete `_Pool`; other delta invoice/checkout/subscription tests
  still use it and will be converted in follow-up slices.
- Stripe remains mocked because the true external boundary is Stripe webhook
  construction, not first-party persistence.

## Deferred

- Convert the remaining delta invoice-paid, subscription-created, checkout, and
  non-delta subscription fake-pool tests.
- Drain or remove `_Pool` only after the remaining fake-pool users are gone.

Parked hardening: none.

## Verification

- Command: python -m py_compile tests/test_atlas_billing_content_ops_deflection_stripe_paid.py
- Command: ATLAS_MIGRATION_TEST_DATABASE_URL=postgresql://atlas:atlas@localhost:5433/atlas pytest tests/test_atlas_billing_content_ops_deflection_stripe_paid.py -k 'delta_invoice_payment_failed'
  - 1 passed, 58 deselected.
- Command: ATLAS_MIGRATION_TEST_DATABASE_URL=postgresql://atlas:atlas@localhost:5433/atlas pytest tests/test_atlas_billing_content_ops_deflection_stripe_paid.py
  - 59 passed.
- Command: python scripts/maturity_sweep.py atlas_brain/api --tests-root tests --top 10
  - Pass/advisory; `atlas_brain/api/billing.py` remains flagged with known
    existing debt (`SWALLOWED_EXCEPT x4`, `INTERNAL_MOCK x38`,
    `UNGUARDED_INDEX x3`).

## Estimated diff size

| File | LOC |
|---|---:|
| `plans/PR-Billing-Delta-Invoice-Payment-Failed-Live-Adapter.md` | 122 |
| `tests/test_atlas_billing_content_ops_deflection_stripe_paid.py` | 117 |
| **Total** | **239** |

# PR-Billing-Delta-Subscription-Updated-Live-Adapter

## Why this slice exists

Issue #1877 is burning down tests that mock first-party billing persistence
instead of exercising the real DB adapter. The Checkout-completion endpoint is
now closed through #1909, so the next fake cluster is the deflection-delta
subscription entitlement webhook path.

Root cause: `test_stripe_webhook_delta_subscription_updated_upserts_entitlement`
uses the hand-written `_Pool` fake and asserts SQL snippets / call order for
`billing_events` instead of proving the webhook persists the billing event
through asyncpg/Postgres. That keeps the entitlement grant path partly real
through `InMemoryDeflectionReportArtifactStore` but leaves the audit/idempotency
boundary fake.

This PR fixes the root for one webhook event (`customer.subscription.updated`)
by replacing `_Pool` with the live billing DB adapter and persisted
`billing_events` assertions. It does not try to drain every remaining delta
event in one PR; those are separate permutations with their own event payloads.

## Scope (this PR)

Ownership lane: real-adapters/test-quality
Slice phase: Production hardening

1. Convert
   `test_stripe_webhook_delta_subscription_updated_upserts_entitlement` from
   `_Pool` to `_connect_live_billing_pool()`.
2. Assert the webhook still grants deflection-delta entitlement through the
   in-memory artifact store while the billing event is persisted in the live
   database with the expected account, event id, and event type.
3. Preserve the prior no-`saas_accounts`-mutation invariant as a live
   before/after row-state assertion.
4. Keep Stripe itself mocked at the webhook boundary; this is a billing
   persistence/adapter proof, not a Stripe API integration test.

### Review Contract

- Acceptance criteria:
  - The converted test has `@pytest.mark.integration`.
  - The converted test no longer instantiates `_Pool` or asserts on
    `execute_calls`.
  - The test applies live billing migrations, account-scoped cleanup, and closes
    the pool in `finally`.
  - The entitlement grant remains observable through
    `list_deflection_delta_entitled_account_ids()`.
  - The test proves `saas_accounts` plan/status/customer/subscription fields do
    not change during a delta subscription entitlement webhook.
  - The billing event is asserted from Postgres, not from fake SQL call capture.
- Affected surfaces:
  - Stripe webhook tests for deflection-delta subscription entitlement.
  - Live billing test helpers in
    `tests/test_atlas_billing_content_ops_deflection_stripe_paid.py`.
- Risk areas:
  - Accidentally widening cleanup beyond the test account/event.
  - Masking the entitlement grant by replacing the artifact store boundary with
    another fake.
  - Leaving the test in the non-integration unit backstop.
- Triggered reviewer rules:
  - R2 Test evidence.
  - R3 Security/auth/payment path.
  - R8 Stateful persistence.
  - R14 Codebase verification.

### Files touched

- `plans/PR-Billing-Delta-Subscription-Updated-Live-Adapter.md`
- `tests/test_atlas_billing_content_ops_deflection_stripe_paid.py`

## Mechanism

- In the converted test:
  - connect to the live migration test DB;
  - apply the billing migrations;
  - delete prior rows for the generated account/event with the existing
    account-scoped live cleanup helper;
  - seed a `saas_accounts` row with account-scoped sentinel
    plan/status/customer/subscription values required by the `billing_events`
    account FK and no-mutation proof;
  - run the webhook with the same mocked Stripe module and in-memory entitlement
    store;
  - assert the entitlement store includes the account;
  - assert the seeded `saas_accounts` fields are unchanged after the webhook;
  - assert `billing_events` contains exactly one row for the delta subscription
    event, account, event type, and JSON payload.
- Close the live pool in a `finally` block.

## Intentional

- The entitlement artifact store remains in-memory because this slice is focused
  on replacing the billing DB fake. Later slices can decide whether to exercise
  the Postgres artifact store for entitlement rows.
- This does not delete `_Pool`; other delta/invoice tests still use it and will
  be converted in follow-up slices.
- Stripe remains mocked because the true external boundary is Stripe webhook
  construction, not first-party persistence.

## Deferred

- Convert the remaining delta invoice/checkout/subscription fake-pool tests.
- Convert `test_non_delta_subscription_does_not_create_delta_entitlement` from
  fake customer-account lookup to live DB.
- Drain or remove `_Pool` only after the remaining fake-pool users are gone.

Parked hardening: none.

## Verification

- py_compile for `tests/test_atlas_billing_content_ops_deflection_stripe_paid.py`
  - Pass.
- Focused live pytest for `tests/test_atlas_billing_content_ops_deflection_stripe_paid.py`
  with `-k 'delta_subscription_updated'`
  - Pass: 1 passed, 58 deselected.
- Full live pytest for `tests/test_atlas_billing_content_ops_deflection_stripe_paid.py`
  - Pass: 59 passed.
- Maturity sweep: `python scripts/maturity_sweep.py atlas_brain/api --tests-root tests --top 10`
  - Pass/advisory: `billing.py` remains score 178 with `INTERNAL_MOCK x38`.

## Estimated diff size

| File | LOC |
|---|---:|
| `plans/PR-Billing-Delta-Subscription-Updated-Live-Adapter.md` | 126 |
| `tests/test_atlas_billing_content_ops_deflection_stripe_paid.py` | 114 |
| **Total** | **240** |

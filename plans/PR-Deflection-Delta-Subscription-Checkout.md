# PR-Deflection-Delta-Subscription-Checkout

## Why this slice exists

#1873 replaces monthly Report Delta manual entitlement with Stripe Billing
subscription state. #1880 made webhook lifecycle events mutate the durable
entitlement table, including `customer.subscription.created`, but ATLAS
checkout creation still rejects the configured monthly delta Price IDs and does
not stamp the created Stripe Subscription with the metadata the webhook mapper
needs.

Root cause: the authenticated `/billing/checkout` allowlist only admits the
generic SaaS/LLM plan Prices, and its Checkout Session metadata is not copied
into `subscription_data.metadata`. For monthly delta subscriptions, that means
ATLAS can have a webhook writer but no ATLAS-owned purchase path that creates a
subscription object with `source=content_ops_deflection_delta_subscription` and
`account_id`.

This PR fixes the root for the checkout entrypoint only: configured monthly
delta Price IDs are valid subscription Checkout prices, and delta checkouts
stamp both the Checkout Session and resulting Subscription metadata so the
signed `customer.subscription.created` webhook can grant the entitlement.

## Scope (this PR)

Ownership lane: deflection/delta-billing-entitlements
Slice phase: Production hardening

1. Allow configured `stripe_content_ops_deflection_delta_price_ids` through the
   existing authenticated Billing Checkout price gate.
2. For those Prices only, add the delta source/account metadata to both
   Checkout Session metadata and `subscription_data.metadata`.
3. Prove the Checkout Session still uses `mode="subscription"`, omits
   `payment_method_types`, and keeps generic plan metadata unchanged.

### Review Contract

- Acceptance criteria:
  - [ ] Blank delta Price config remains fail-closed; arbitrary unconfigured
        Prices are still rejected when any allowed Prices are configured.
  - [ ] A configured delta Price creates a subscription Checkout Session with
        `metadata.source` and `subscription_data.metadata.source` set to
        `content_ops_deflection_delta_subscription`, using the same source
        constant as the webhook detector.
  - [ ] The subscription metadata includes the authenticated ATLAS account id so
        `customer.subscription.created` can map back without guessing.
  - [ ] The Checkout call still omits `payment_method_types`; Stripe dynamic
        payment methods remain enabled.
  - [ ] Existing generic plan checkout behavior and idempotency key derivation
        remain unchanged.
- Affected surfaces: `atlas_brain/api/billing.py` and focused billing tests.
- Risk areas: Stripe Checkout, subscription lifecycle entitlement grants,
  payment metadata trust boundary, account mapping.
- Reviewer rules triggered: R1, R2, R3, R5, R8, R10, R11, R13, R14.
- boundary-probe: real `POST /billing/checkout` with a configured delta Price
  stamps webhook-detectable subscription metadata; unconfigured Price rejected;
  generic growth checkout has no delta source metadata.

### Files touched

- `atlas_brain/api/billing.py`
- `plans/PR-Deflection-Delta-Subscription-Checkout.md`
- `tests/test_atlas_billing_stripe_hardening.py`

## Mechanism

`create_checkout` reuses the existing monthly delta Price parser from the
webhook path and includes those Prices in its allowed-price set.

When the selected price is a delta subscription Price, the Checkout Session
metadata becomes:

```python
{
    "account_id": account_id,
    "source": DEFLECTION_DELTA_SUBSCRIPTION_SOURCE,
}
```

and the same mapping is passed as `subscription_data.metadata`. The latter is
the load-bearing part: Stripe sends that metadata on
`customer.subscription.created`, and #1880's webhook handler reads the
subscription object directly.

Generic plan checkouts keep the existing metadata shape and idempotency key.
The source value is a single billing constant shared by the Checkout producer
and the webhook detector so producer/consumer drift is caught by the endpoint
regression.

## Intentional

- No new public Content Ops route is added in this slice. The existing
  authenticated billing checkout already owns subscription Checkout Session
  creation; portfolio/browser wiring remains a later consumer.
- No `payment_method_types` is added. Stripe dynamic payment methods stay
  dashboard-controlled.
- No customer portal UX change is included; the existing `/billing/portal`
  endpoint remains the management surface once a Stripe customer exists.

## Deferred

- Portfolio/results-page upsell wiring is deferred until the frontend lane is
  ready to call the authenticated billing checkout path.
- Customer Portal copy/UX for monthly Report Deltas remains deferred.
- Historical subscription backfill remains deferred; this only affects new
  Checkout-created subscriptions.

Parked hardening: none.

## Verification

- Command: python -m py_compile atlas_brain/api/billing.py tests/test_atlas_billing_stripe_hardening.py - passed.
- Command: python -m pytest tests/test_atlas_billing_stripe_hardening.py -q - passed, 9 passed, 1 warning.
- Command: python scripts/maturity_sweep.py atlas_brain/api --tests-root tests --baseline tests/maturity_sweep/baseline_atlas_brain_api.json --min-score 8 --sensitive-glob '**/billing/**' --sensitive-glob '**/billing*' --sensitive-glob '**/paid*' --sensitive-glob '**/auth/**' --sensitive-glob '**/auth*' --sensitive-glob '**/webhook*' --sensitive-glob '**/webhooks/**' --sensitive-glob '**/payment*' --sensitive-glob '**/invoicing/**' --sensitive-glob '**/*invoice*' --sensitive-glob '**/*deletion*' - passed, no new brittleness above baseline.
- Pending before push: local PR review through `scripts/push_pr.sh`.

## Estimated diff size

| File | LOC |
|---|---:|
| `atlas_brain/api/billing.py` | 17 |
| `plans/PR-Deflection-Delta-Subscription-Checkout.md` | 124 |
| `tests/test_atlas_billing_stripe_hardening.py` | 164 |
| **Total** | **305** |

# PR-Deflection-Delta-Stripe-Entitlement-Webhooks

## Why this slice exists

#1873's durable target is "Stripe Billing subscriptions are the source of truth
for monthly deflection delta delivery." #1876 added the fail-closed resolver and
table, but nothing writes Stripe lifecycle state into that table yet. The
monthly task can now read Billing-backed entitlements, but Stripe webhooks still
only update generic SaaS plan state and one-time paid report access.

Root cause: the deflection delta subscription product has no configured Price-ID
gate and no webhook producer for `content_ops_deflection_delta_entitlements`.
That leaves monthly deltas dependent on the transitional config allowlist, and a
future delta subscription event would fall through the generic subscription
handler instead of preserving the one-time-report/monthly-delta boundary.

This PR fixes the root for the first Stripe lifecycle vertical: configured
monthly delta Price IDs produce ATLAS-owned entitlement rows, and inactive
Stripe states fail closed by revoking those rows.

Review-finding root cause: the first push recorded lifecycle state but did not
persist any monotonic Stripe event basis, so a later-delivered older `active`
event could overwrite a newer revoke. The same first pass also proved routing
with fake-pool SQL assertions instead of a handler-level store contract. This
fixes that root by carrying Stripe `event.created` into the entitlement row,
ignoring older lifecycle updates on conflict, and proving the lifecycle handler
against the real in-memory store/resolver.

Round-2 review root cause: the webhook fixture covered the legacy invoice shape
instead of the configured `2026-05-27.dahlia` shape, so renewal/dunning invoice
events could miss both Price ID and subscription ID extraction. The recency
guard also treated a missing incoming event timestamp as newest. This update
adds the dahlia invoice paths, routes configured delta
`customer.subscription.created`, and requires an incoming numeric event
timestamp before an existing entitlement row can be overwritten.

Diff-size note: this exceeds the 400 LOC soft cap because the smallest safe
root fix crosses the billing webhook and the entitlement store boundary, and the
tests intentionally cover the real adapter contract plus the webhook routing
matrix that prevents one-time and generic subscription flows from merging.
Splitting out the tests or store adapter would recreate the fake-transport gap
that #1876 just closed.

## Scope (this PR)

Ownership lane: deflection/delta-billing-entitlements
Slice phase: Production hardening

1. Add a typed config field for the monthly deflection delta Stripe Price IDs
   and a billing helper that treats only those configured Prices as delta
   subscription products.
2. Add store-level entitlement write/revoke methods and route Stripe
   subscription lifecycle events through the real store adapter, not hand-coded
   webhook SQL.
3. Handle `checkout.session.completed`, `customer.subscription.created`,
   `customer.subscription.updated`, `customer.subscription.deleted`,
   `invoice.paid`, and
   `invoice.payment_failed` only when the Stripe object contains a configured
   monthly delta Price ID.
4. Prove active/trialing rows grant and past_due/unpaid/canceled/deleted rows
   revoke/fail closed without touching one-time paid report access.

### Review Contract

- Acceptance criteria:
  - [ ] Configured delta Price IDs are the only path that can create/update
        `content_ops_deflection_delta_entitlements`.
  - [ ] Unconfigured subscription Prices do not create a delta entitlement and
        continue through the existing generic SaaS subscription path.
  - [ ] Active/trialing delta subscription lifecycle events write
        `revoked_at = NULL`; past_due/unpaid/canceled/incomplete/paused/deleted
        lifecycle events set `revoked_at` and therefore fail closed under the
        #1876 resolver.
  - [ ] Out-of-order Stripe redelivery cannot revive a revoked subscription:
        lifecycle writes carry Stripe `event.created`, and older events lose to
        newer entitlement rows; missing incoming event timestamps cannot
        overwrite existing timestamped rows.
  - [ ] Dahlia invoice payloads resolve delta Price IDs from
        `lines.data[].pricing.price_details.price` and subscription IDs from
        `parent.subscription_details.subscription`.
  - [ ] `customer.subscription.created` grants active/trialing configured
        delta subscriptions without broadening generic subscription-created
        handling.
  - [ ] One-time paid deflection report checkout/refund/dispute behavior is
        unchanged.
  - [ ] Tests use the real entitlement store adapter contract for the DB write
        behavior; fakes may mock transport, but must not reimplement the
        entitlement predicate.
- Affected surfaces: `atlas_brain/api/billing.py`,
  `atlas_brain/config.py`, `extracted_content_pipeline/deflection_report_access.py`,
  and their billing/adapter tests.
- Risk areas: billing/auth/payment lifecycle, fail-closed entitlement state,
  Stripe webhook idempotency, and config misclassification.
- Reviewer rules triggered: R1, R2, R3, R5, R7, R8, R10, R11, R12, R13, R14.
- boundary-probe: negative tests for unconfigured Price IDs, inactive/revoked
  statuses, missing account/customer mapping, and a one-time report checkout
  event that must not create a delta entitlement.

### Files touched

- `atlas_brain/api/billing.py`
- `atlas_brain/config.py`
- `extracted_content_pipeline/deflection_report_access.py`
- `plans/PR-Deflection-Delta-Stripe-Entitlement-Webhooks.md`
- `tests/test_atlas_billing_content_ops_deflection_stripe_paid.py`
- `tests/test_content_ops_deflection_delta_persistence.py`

## Mechanism

`SaaSAuthConfig` gains a comma-separated
`ATLAS_SAAS_STRIPE_CONTENT_OPS_DEFLECTION_DELTA_PRICE_IDS` setting. Billing
parses it into an exact Price-ID set; blank means no Stripe subscription can
grant monthly deltas.

`DeflectionReportArtifactStore` gets explicit entitlement lifecycle methods.
The Postgres implementation upserts by `stripe_subscription_id` into
`content_ops_deflection_delta_entitlements`, stores customer/price/status/period
metadata, clears `revoked_at` for active/trialing, and sets `revoked_at` for
non-granting statuses. Stripe webhook handlers pass the signed event's
`created` timestamp into the write, and the upsert only updates an existing row
when the incoming event timestamp is present and at least as new as the stored
timestamp.
The in-memory implementation mirrors that state so the resolver and writer are
tested as one adapter contract.

`billing.py` extracts subscription Price IDs from Stripe objects and only
handles monthly delta entitlement writes when one of those IDs matches the
configured allowlist. For matching subscription events, billing resolves the
ATLAS account from subscription metadata or Stripe customer mapping, then calls
`PostgresDeflectionReportArtifactStore(pool)` to upsert/revoke. Non-delta
subscription events keep the existing generic SaaS plan behavior.

Invoice paid/failed events use the invoice's subscription/line Price IDs when
present, including dahlia `pricing.price_details.price` and
`parent.subscription_details.subscription`, so a payment recovery can regrant
and a failed renewal can revoke before the next subscription update arrives. If
the invoice lacks enough subscription identity to prove it is the configured
delta product, it does not write an entitlement.

## Intentional

- No Stripe API fetch is added in this slice. The webhook must be deterministic
  from the signed event payload and existing ATLAS account mapping; missing
  subscription/price/account data fails closed instead of guessing.
- The transitional `ATLAS_DEFLECTION_DELTA_ENTITLED_ACCOUNT_IDS` allowlist
  remains as a fallback for accounts with no Billing row yet. #1876 made known
  inactive/revoked Billing rows suppress that fallback.
- The generic SaaS plan subscription handler remains in place for non-delta
  products; this PR only carves out the monthly deflection delta product so it
  cannot mutate unrelated plan state.

## Deferred

- Stripe Checkout creation for the delta subscription landing/upsell path is
  deferred. This slice only consumes signed lifecycle events once the product
  and Price IDs are configured.
- Customer Portal UX and go-live rehearsal remain deferred to the #1873
  operational checklist.
- Event backfill for historical subscriptions is deferred; this slice is
  going-forward lifecycle wiring.

Parked hardening: none.

## Verification

- Command: python -m py_compile atlas_brain/api/billing.py atlas_brain/config.py extracted_content_pipeline/deflection_report_access.py tests/test_content_ops_deflection_delta_persistence.py tests/test_atlas_billing_content_ops_deflection_stripe_paid.py - passed.
- Command: python -m pytest tests/test_content_ops_deflection_delta_persistence.py tests/test_atlas_billing_content_ops_deflection_stripe_paid.py -q - passed, 92 passed, 1 warning.
- Command: python scripts/maturity_sweep.py atlas_brain/api --tests-root tests --baseline tests/maturity_sweep/baseline_atlas_brain_api.json --min-score 8 --sensitive-glob '**/billing/**' --sensitive-glob '**/billing*' --sensitive-glob '**/paid*' --sensitive-glob '**/auth/**' --sensitive-glob '**/auth*' --sensitive-glob '**/webhook*' --sensitive-glob '**/webhooks/**' --sensitive-glob '**/payment*' --sensitive-glob '**/invoicing/**' --sensitive-glob '**/*invoice*' --sensitive-glob '**/*deletion*' - passed, no new brittleness above baseline.

## Estimated diff size

| File | LOC |
|---|---:|
| `atlas_brain/api/billing.py` | 295 |
| `atlas_brain/config.py` | 7 |
| `extracted_content_pipeline/deflection_report_access.py` | 189 |
| `plans/PR-Deflection-Delta-Stripe-Entitlement-Webhooks.md` | 180 |
| `tests/test_atlas_billing_content_ops_deflection_stripe_paid.py` | 537 |
| `tests/test_content_ops_deflection_delta_persistence.py` | 125 |
| **Total** | **1333** |

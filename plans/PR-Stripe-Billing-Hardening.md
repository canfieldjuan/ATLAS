# PR-Stripe-Billing-Hardening

## Why this slice exists

The Atlas billing Stripe integration currently mutates the global Stripe module
with the configured key and relies on the account/default API version. Review
flagged three moderate hardening gaps: API version drift, using a full `sk_`
secret where a restricted `rk_` key is preferred, and missing idempotency keys
on billing-side Customer and Checkout creation.

This slice closes the code-enforceable pieces in `atlas_brain/api/billing.py`
without changing the public billing endpoints or Stripe product contracts. A
review follow-up also migrates the adjacent in-tree Stripe callers that share
the process-wide Stripe module so the billing API-version pin cannot leak into
routes that still only reset `stripe.api_key`.

## Scope (this PR)

Ownership lane: atlas-billing/stripe
Slice phase: Production hardening

1. Centralize billing Stripe SDK setup in `billing.py` and pin the Stripe API
   version used by billing calls.
2. Keep the existing `ATLAS_SAAS_STRIPE_SECRET_KEY` config field but log a
   warning when it looks like a full `sk_` key instead of the preferred
   restricted `rk_` key.
3. Add deterministic idempotency keys to `stripe.Customer.create` and
   `stripe.checkout.Session.create` in the authenticated billing checkout
   route.
4. Apply the same Stripe SDK setup helper to adjacent auth/vendor Stripe callers
   that would otherwise inherit the process-wide billing API version.
5. Add focused tests for API-version pinning, restricted-key warning behavior,
   the idempotency key values passed to Stripe, and vendor checkout/session
   version pinning.

### Files touched

- `plans/PR-Stripe-Billing-Hardening.md`
- `.github/workflows/atlas_content_ops_deflection_stripe_paid_checks.yml`
- `atlas_brain/api/auth.py`
- `atlas_brain/api/b2b_vendor_briefing.py`
- `atlas_brain/api/billing.py`
- `tests/test_atlas_billing_content_ops_deflection_stripe_paid.py`
- `tests/test_atlas_billing_stripe_hardening.py`
- `tests/test_b2b_vendor_briefing.py`

## Mechanism

`_get_stripe()` delegates to a small `_configure_stripe_module()` helper:

```text
stripe.api_key = cfg.stripe_secret_key
stripe.api_version = STRIPE_API_VERSION
```

The same helper is used by the webhook route before signature verification.
The configured key field is unchanged, so operators can replace an `sk_` with
an `rk_` without a config migration. If an `sk_` key is still present, the code
logs a warning naming the restricted-key expectation without logging the key.

Checkout creation gains stable Stripe idempotency keys:

```text
atlas-customer:<account_id>
atlas-checkout:<account_id>:<user_id>:<sha256(price_id/success_url/cancel_url)>
```

Those keys are deterministic for a retry/double-submit of the same operation
and do not contain Stripe secrets or raw URLs.

Because `stripe.api_version` is process-wide on the imported Stripe module,
the same helper is also used in the adjacent signup and vendor-briefing routes
that were previously resetting only `stripe.api_key`. That keeps all in-tree
module-level Stripe usage on the pinned version instead of letting billing
requests change another route's version implicitly.

## Intentional

- This does not fail startup on existing `sk_` keys. The safer `rk_` key is an
  operator secret rotation through the same env field; warning first avoids
  breaking current deployments during the hardening pass.
- The webhook event payload shape is ultimately controlled by the Stripe
  webhook endpoint API version in Stripe's dashboard. This PR pins the SDK
  version used by Atlas billing code and documents the expected version in one
  constant, but an operator still must pin the dashboard endpoint to match.
- This does not touch portfolio-ui Stripe Checkout; that code uses direct
  Stripe HTTP calls and is outside module-level Stripe SDK usage.
- Adjacent auth/vendor Stripe callers are migrated only to prevent the
  process-wide billing API-version pin from leaking across routes. Their own
  endpoint semantics and idempotency behavior are otherwise left unchanged.

## Deferred

- Add idempotency discipline to adjacent non-billing Stripe create calls in a
  follow-up PR if those routes stay in production use.
- Add Stripe dashboard/runbook verification for webhook endpoint API-version
  pinning and restricted-key permissions.
- Optional webhook IP allowlisting remains deferred; signature verification is
  still the primary control.
- Parked hardening: none.

## Verification

- `python -m py_compile atlas_brain/api/billing.py atlas_brain/api/auth.py atlas_brain/api/b2b_vendor_briefing.py tests/test_atlas_billing_stripe_hardening.py tests/test_b2b_vendor_briefing.py` -- passed.
- `python -m pytest tests/test_atlas_billing_stripe_hardening.py -q` -- 3 passed, 1 warning.
- `python -m pytest tests/test_b2b_vendor_briefing.py -q` -- 41 passed, 1 warning.
- `python -m pytest tests/test_atlas_billing_stripe_hardening.py tests/test_atlas_billing_content_ops_deflection_stripe_paid.py tests/test_atlas_billing_content_ops_deflection_paid_flow.py -q` -- 22 passed, 1 warning.
- `python -m pytest tests/test_atlas_billing_stripe_hardening.py tests/test_b2b_vendor_briefing.py tests/test_atlas_billing_content_ops_deflection_stripe_paid.py tests/test_atlas_billing_content_ops_deflection_paid_flow.py -q` -- 63 passed, 1 warning.
- `bash scripts/local_pr_review.sh --current-pr-body-file /home/juan-canfield/Desktop/atlas-pr-bodies/stripe-billing-hardening.md` -- passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan doc | 123 |
| Workflow enrollment | 10 |
| Adjacent Stripe caller setup | 9 |
| Billing helper/idempotency code | 48 |
| Existing webhook assertion | 1 |
| Focused tests | 206 |
| **Total** | **397** |

Actual diff is 8 files, +390 / -7. Under the 400 LOC soft cap.

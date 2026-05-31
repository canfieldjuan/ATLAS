# PR-FAQ-Deflection-Portfolio-Checkout-RAK

## Why this slice exists

Issue #1161's latest Stripe hardening note calls out that the portfolio
Checkout route should not rely on a full shared Stripe secret key for
production buyer traffic. The portfolio is a separate service that only needs
to create Checkout Sessions, so the safer production posture is a restricted
API key with Checkout write scope.

This slice keeps the already-verified Checkout contract intact while making the
server route prefer `ATLAS_SAAS_STRIPE_RAK` and fail closed on live full-secret
fallbacks.

## Scope (this PR)

Ownership lane: portfolio-ui/faq-deflection
Slice phase: Production hardening

1. Add a small portfolio Checkout key selector that prefers
   `ATLAS_SAAS_STRIPE_RAK`.
2. Keep the existing `STRIPE_SECRET_KEY` preview/test fallback, but reject
   `sk_live_` full secret keys so production cannot silently use the broad key.
3. Use the selected key for configured Price validation and Checkout Session
   creation without exposing key material in responses.
4. Keep configured Price preflight validation on preview/full-key fallback, but
   skip Stripe Price reads when the selected key is a checkout-only RAK.
5. Add focused tests for RAK preference, test-secret fallback, live-secret
   rejection, and checkout-only RAK compatibility with configured Prices.

### Files touched

- `plans/PR-FAQ-Deflection-Portfolio-Checkout-RAK.md`
- `portfolio-ui/api/content-ops/deflection/checkout.js`
- `portfolio-ui/scripts/faq-deflection-result-page.test.mjs`

## Mechanism

The route resolves Stripe credentials before request parsing:

```text
ATLAS_SAAS_STRIPE_RAK (must start with rk_) -> preferred
STRIPE_SECRET_KEY / ATLAS_SAAS_STRIPE_SECRET_KEY -> preview fallback
sk_live_ fallback -> rejected
```

The returned config carries only the selected key in memory and a generic error
code for client responses. `stripeAuthHeaders(...)` still creates the Stripe
Basic auth header server-side for the same direct Checkout HTTP call. The body
contract, metadata, one-time amount floor, and success/cancel URL behavior are
unchanged.

When a configured Stripe Price is used with a checkout-only RAK, the route skips
the preflight `GET /v1/prices/{id}` because that read is outside the key's
intended scope. Preview/full-key fallback still runs the existing Price
validation, and ATLAS still enforces `amount_total >= 150000`, `usd`, and
metadata on the signed webhook before unlocking the report.

## Intentional

- This does not add Stripe SDK dependencies; the route already uses direct
  Stripe HTTP calls and the hardening only changes credential selection.
- Test-mode `sk_test_` fallback remains allowed so previews can keep working
  until the restricted test key is provisioned.
- `sk_live_` is rejected rather than warned because production fallback to a
  full secret key is exactly the blast-radius issue this slice closes.
- Checkout-only RAKs do not read configured Prices before creating Checkout.
  A mispriced session still fails to unlock at the ATLAS webhook amount floor;
  this keeps the route compatible with the least-privilege key.
- This does not touch the open submit/configured-account PR #1204.

## Deferred

- Provisioning the restricted key in Stripe and Vercel env is an operator
  handoff; the PR only enforces the expected runtime posture.
- A live Stripe Checkout smoke with the restricted key remains deferred until
  the key exists in the deployed environment.
- Parked hardening: none.

## Verification

- `node --check portfolio-ui/api/content-ops/deflection/checkout.js && node --check portfolio-ui/scripts/faq-deflection-result-page.test.mjs` -- passed.
- `npm run test:deflection-result --prefix portfolio-ui` -- 15 checks passed.
- `npm run test:deflection-upload-shell --prefix portfolio-ui` -- 19 checks passed.
- `npm run test:deflection-atlas-proxy --prefix portfolio-ui` -- 14 checks passed.
- `bash scripts/local_pr_review.sh --current-pr-body-file /home/juan-canfield/Desktop/atlas-pr-bodies/faq-deflection-portfolio-checkout-rak.md` -- passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan doc | 97 |
| Checkout key selector | 38 |
| Focused tests | 145 |
| **Total** | **280** |

Actual diff is 3 files, +272 / -8. Under the 400 LOC soft cap.

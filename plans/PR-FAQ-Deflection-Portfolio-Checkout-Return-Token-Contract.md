# PR-FAQ-Deflection-Portfolio-Checkout-Return-Token-Contract

## Why this slice exists

PR-FAQ-Deflection-Portfolio-Checkout-Cancel-State added a hosted result-page
notice for cancelled Checkout returns, but review caught that the first version
tested `checkout=cancelled` while the Checkout endpoint actually emits
`checkout=cancel`. The product code was corrected in that PR, but the gap was a
cross-module contract problem: the result-page test used a hand-written token
instead of deriving the token from the Checkout URL builder.

This slice locks that contract so the Checkout endpoint and hosted result-page
renderer cannot silently drift again.

## Scope (this PR)

Ownership lane: portfolio-ui/faq-deflection
Slice phase: Robust testing

1. Add a focused cross-module fixture in the portfolio result-page test.
2. Derive the cancel return token from `checkoutUrls(...)`, not a hard-coded
   assumption.
3. Render the hosted result page with the derived token and assert the cancelled
   state appears without starting the success-only artifact retry.
4. Isolate that fixture from configured Checkout URL override env vars so it
   always verifies the default result-path branch.
5. Leave product behavior unchanged.

### Files touched

- `plans/PR-FAQ-Deflection-Portfolio-Checkout-Return-Token-Contract.md`
- `portfolio-ui/scripts/faq-deflection-result-page.test.mjs`

## Mechanism

The fixture calls `checkoutUrls(...)` with a representative public host and
temporarily clears `DEFLECTION_CHECKOUT_SUCCESS_URL` and
`DEFLECTION_CHECKOUT_CANCEL_URL` so environment overrides cannot mask the
default URL-building branch. It extracts the `checkout` search parameter from
the generated `cancelUrl`, then passes that value to `renderResultPage(...)` and
verifies:

```text
cancelUrl token -> hosted cancellation notice
cancelUrl token -> no artifact retry script
cancelUrl token -> normal Checkout CTA remains available
```

This makes the source of truth the same function the live Checkout handler uses
to create Stripe's `cancel_url`.

## Intentional

- This is test-only. The current product behavior is already correct after
  PR-FAQ-Deflection-Portfolio-Checkout-Cancel-State.
- The fixture derives only the return token; it does not call Stripe or ATLAS.
- The fixture saves and restores Checkout URL override env vars locally; this
  avoids test-environment coupling without changing runtime override behavior.
- The test keeps paid rendering fail-closed by asserting no paid report marker
  appears for the cancelled locked state.

## Deferred

- Broader browser-level Checkout return verification remains deferred until the
  deployed Stripe configuration and paid test artifact are available.
- Parked hardening: none.

## Verification

- `npm run test:deflection-result --prefix portfolio-ui` - passed, 12 checks.
- `DEFLECTION_CHECKOUT_SUCCESS_URL='https://override.example.com/success' DEFLECTION_CHECKOUT_CANCEL_URL='https://override.example.com/cancel' npm run test:deflection-result --prefix portfolio-ui` -
  passed, 12 checks.
- `npm run build --prefix portfolio-ui` - passed using the existing ignored root
  `portfolio-ui/node_modules` install for local verification; `npm ci` in this
  worktree is blocked by pre-existing portfolio lockfile drift.
- `bash scripts/run_extracted_pipeline_checks.sh` - extracted_reasoning_core
  295 passed; extracted_content_pipeline 2864 passed, 10 skipped, 1 warning.
- `bash scripts/local_pr_review.sh --current-pr-body-file /home/juan-canfield/Desktop/atlas-pr-bodies/faq-deflection-portfolio-checkout-return-token-contract.md` -
  passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan doc | ~80 |
| Cross-module fixture | ~55 |
| **Total** | **~135** |

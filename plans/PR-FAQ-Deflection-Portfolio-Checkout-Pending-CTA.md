# PR-FAQ-Deflection-Portfolio-Checkout-Pending-CTA

## Why this slice exists

PR-FAQ-Deflection-Portfolio-Checkout-Success-Retry added a bounded retry after
Stripe redirects the buyer back with `checkout=success` while ATLAS is still
waiting on the verified webhook. That closed the refresh gap, but the hosted
result page still labels the side panel as "Unlock full report" and leaves the
Checkout button enabled while the retry is active.

That state can invite a second Checkout session during the webhook race. The
paid boundary remains safe because ATLAS still unlocks only from the verified
webhook, but the customer experience should clearly communicate that payment is
being processed and avoid duplicate Checkout starts.

## Scope (this PR)

Ownership lane: portfolio-ui/faq-deflection
Slice phase: Production hardening

1. Treat `checkout=success` plus `artifact_status: "locked"` as a pending
   payment-processing state on the hosted result page.
2. Disable the Checkout CTA while the pending unlock retry is active.
3. Update the side-panel heading/copy/button label for that pending state.
4. Keep the existing bounded artifact-status retry and server-rendered paid
   artifact boundary unchanged.
5. Add focused tests that pin the pending CTA and the unlocked state.

### Files touched

- `plans/PR-FAQ-Deflection-Portfolio-Checkout-Pending-CTA.md`
- `portfolio-ui/api/content-ops/deflection/result-page.js`
- `portfolio-ui/scripts/faq-deflection-atlas-proxy.test.mjs`

## Mechanism

`renderResultPage(...)` already derives `shouldRetryArtifact` from:

```text
checkoutStatus === "success" && artifactStatus === "locked"
```

This slice reuses that predicate for the CTA state. When true, the side panel
renders a pending heading/copy/button label and the button is disabled. The
existing retry script still polls the same-origin report proxy and reloads only
after `artifact_status === "unlocked"`. Once the hosted page reloads with an
unlocked artifact, the existing unlocked state takes over.

## Intentional

- The pending state is UI hardening only. It does not change Checkout creation,
  Stripe webhook verification, artifact fetches, or paid artifact rendering.
- The button is disabled only for the success-return locked state and the
  already-unlocked state. A normal locked report without `checkout=success`
  still starts Checkout.
- The browser still does not render paid artifact payloads from the retry JSON.

## Deferred

- Richer progress messaging or manual retry controls remain deferred until the
  live paid-return path has production observations.
- Parked hardening: none.

## Verification

- `npm run test:deflection-atlas-proxy --prefix portfolio-ui` - passed, 13 checks.
- `npm run test:deflection-result --prefix portfolio-ui` - passed, 11 checks.
- `npm run build --prefix portfolio-ui` - passed using the existing ignored root
  `portfolio-ui/node_modules` install for local verification; `npm ci` in this
  worktree is blocked by pre-existing portfolio lockfile drift.
- `bash scripts/run_extracted_pipeline_checks.sh` - extracted_reasoning_core
  295 passed; extracted_content_pipeline 2864 passed, 10 skipped, 1 warning.
- `bash scripts/local_pr_review.sh --current-pr-body-file /home/juan-canfield/Desktop/atlas-pr-bodies/faq-deflection-portfolio-checkout-pending-cta.md` -
  passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan doc | ~80 |
| Hosted CTA state | ~20 |
| Tests | ~25 |
| **Total** | **~125** |

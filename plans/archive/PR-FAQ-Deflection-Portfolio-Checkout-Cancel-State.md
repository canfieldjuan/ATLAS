# PR-FAQ-Deflection-Portfolio-Checkout-Cancel-State

## Why this slice exists

The hosted FAQ deflection result page now handles the successful Checkout return
race: it retries artifact status while ATLAS waits for the verified Stripe
webhook and disables duplicate Checkout starts during that pending state. The
remaining Checkout return branch is cancellation. Today
`checkout=cancel` only appears in the metadata table, so the buyer lands back
on a locked report without an explicit explanation or next action.

This slice makes cancellation explicit without changing payment boundaries. A
cancelled Checkout return stays locked, shows a clear notice, and keeps the
normal Checkout CTA available so the buyer can restart intentionally.

## Scope (this PR)

Ownership lane: portfolio-ui/faq-deflection
Slice phase: Product polish

1. Render a cancellation notice when the hosted result page receives the
   existing Checkout return token, `checkout=cancel`.
2. Keep `checkout=success` behavior and the pending webhook retry unchanged.
3. Keep the normal locked-report Checkout CTA enabled after cancellation.
4. Add focused tests for the cancel notice and non-retry behavior.

### Files touched

- `plans/PR-FAQ-Deflection-Portfolio-Checkout-Cancel-State.md`
- `portfolio-ui/api/content-ops/deflection/result-page.js`
- `portfolio-ui/scripts/faq-deflection-atlas-proxy.test.mjs`

## Mechanism

`renderResultPage(...)` already receives the sanitized `checkoutStatus` query
value. This slice extends the existing status banner decision:

```text
checkoutStatus === "success"   -> success/webhook notice
checkoutStatus === "cancel"     -> cancellation notice
otherwise                      -> no banner
```

No paid artifact state changes derive from the query string. The artifact
status still comes from the server-side ATLAS proxy, and the retry script still
emits only for `checkout=success` plus `artifact_status: "locked"`.

## Intentional

- The cancelled state does not call Stripe, ATLAS `/paid`, or `/artifact`
  differently. It only explains the return state.
- The Checkout CTA remains enabled after cancellation because no payment has
  been confirmed and no webhook race is expected.
- The browser still cannot create paid report content from a query parameter or
  snapshot field.

## Deferred

- Richer cancelled-checkout analytics or abandoned-checkout follow-up remains
  deferred until this flow has production observations.
- Parked hardening: none.

## Verification

- `npm run test:deflection-atlas-proxy --prefix portfolio-ui` - passed, 14 checks.
- `npm run test:deflection-result --prefix portfolio-ui` - passed, 11 checks.
- `npm run build --prefix portfolio-ui` - passed using the existing ignored root
  `portfolio-ui/node_modules` install for local verification; `npm ci` in this
  worktree is blocked by pre-existing portfolio lockfile drift.
- `bash scripts/run_extracted_pipeline_checks.sh` - extracted_reasoning_core
  295 passed; extracted_content_pipeline 2864 passed, 10 skipped, 1 warning.
- `bash scripts/local_pr_review.sh --current-pr-body-file /home/juan-canfield/Desktop/atlas-pr-bodies/faq-deflection-portfolio-checkout-cancel-state.md` -
  passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan doc | ~75 |
| Hosted cancel notice | ~10 |
| Tests | ~20 |
| **Total** | **~105** |

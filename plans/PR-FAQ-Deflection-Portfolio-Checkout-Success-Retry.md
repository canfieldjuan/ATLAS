# PR-FAQ-Deflection-Portfolio-Checkout-Success-Retry

## Why this slice exists

The hosted portfolio result page now fetches ATLAS snapshot and artifact state
server-side, renders paid artifact Markdown only after `artifact_status:
"unlocked"`, and starts Stripe Checkout from the customer path. Issue #1161
called out the remaining customer-flow race: after Checkout redirects back with
`checkout=success`, the browser can arrive before ATLAS receives and verifies
Stripe's webhook. The current hosted page probes `/artifact` once during server
render, then leaves the customer on the locked snapshot view until they manually
refresh.

This slice adds a bounded success-return retry that reuses the existing public
portfolio report proxy. It improves the paid-return handoff without weakening
the paywall trust boundary: the browser only polls for `artifact_status` and
reloads the hosted page after ATLAS reports the paid artifact unlocked.

## Scope (this PR)

Ownership lane: portfolio-ui/faq-deflection
Slice phase: Production hardening

1. Detect the `checkout=success` plus locked-artifact state on the hosted result
   page.
2. Add a bounded browser retry loop that polls
   `/api/content-ops/deflection/report` for the same `request_id` and
   `account_id`.
3. Reload the hosted result page only after the proxy reports
   `artifact_status: "unlocked"`.
4. Keep paid artifact rendering exclusively owned by the server-rendered hosted
   page after ATLAS unlocks the artifact.
5. Add focused tests for the retry script, non-success checkout state, and the
   no-paid-copy boundary.

### Files touched

- `plans/PR-FAQ-Deflection-Portfolio-Checkout-Success-Retry.md`
- `portfolio-ui/api/content-ops/deflection/result-page.js`
- `portfolio-ui/scripts/faq-deflection-atlas-proxy.test.mjs`

## Mechanism

`renderResultPage(...)` already receives the server-side proxy envelope. This
slice derives:

```text
checkoutStatus === "success" && artifactStatus === "locked"
```

When that predicate is true, the hosted page emits a small retry script. The
script polls the existing same-origin report proxy with the validated
`request_id` and `account_id`, waits between attempts, and calls
`window.location.reload()` only when the JSON envelope says
`artifact_status === "unlocked"`. The script does not read or render
`artifact.markdown`; the reload path lets the existing server renderer fetch
and escape the paid artifact.

## Intentional

- The retry is bounded. It is not an open-ended poller and does not turn the
  hosted result page into an async job UI.
- The browser does not render paid content from the JSON proxy. It checks only
  the status field, then reloads for the server-rendered paid artifact path.
- Missing or failed artifact states do not retry. A `checkout=success` return
  with `artifact_status: "missing"` is more likely a wrong request/account than
  a webhook race.
- This does not change Stripe Checkout creation or ATLAS webhook handling.

## Deferred

- Richer progress messaging or manual retry controls can be added after the
  live paid-return path is observed in production.
- Parked hardening: none.

## Verification

- `npm run test:deflection-atlas-proxy --prefix portfolio-ui` - passed, 13 checks.
- `npm run test:deflection-result --prefix portfolio-ui` - passed, 11 checks.
- `npm run build --prefix portfolio-ui` - passed using the existing ignored root
  `portfolio-ui/node_modules` install for local verification; `npm ci` in this
  worktree is blocked by pre-existing portfolio lockfile drift.
- `bash scripts/run_extracted_pipeline_checks.sh` - extracted_reasoning_core
  295 passed; extracted_content_pipeline 2864 passed, 10 skipped, 1 warning.
- `bash scripts/local_pr_review.sh --current-pr-body-file /home/juan-canfield/Desktop/atlas-pr-bodies/faq-deflection-portfolio-checkout-success-retry.md` -
  passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan doc | ~85 |
| Hosted retry script | ~55 |
| Tests | ~35 |
| **Total** | **~175** |

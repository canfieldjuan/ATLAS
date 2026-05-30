# PR-FAQ-Deflection-Portfolio-Paid-Report-Renderer

## Why this slice exists

PR-FAQ-Deflection-Portfolio-Atlas-Proxy gave the hosted portfolio result page a
server-only ATLAS proxy and real snapshot hydration, then intentionally deferred
paid artifact rendering. The next customer-visible step is to let the hosted
result page display the full `FAQDeflectionReportArtifact` after ATLAS returns
an unlocked artifact envelope.

This slice keeps the paywall trust boundary intact: locked reports remain
snapshot-only, and the hosted page never derives paid copy from free snapshot
fields. The paid report appears only when the portfolio server receives
`artifact_status: "unlocked"` with artifact Markdown from ATLAS.

## Scope (this PR)

Ownership lane: portfolio-ui/faq-deflection
Slice phase: Vertical slice

1. Render the paid artifact Markdown on the hosted result page only for an
   unlocked ATLAS artifact envelope.
2. Escape the Markdown/HTML as text instead of injecting rendered HTML.
3. Keep locked, missing, invalid, or unavailable artifact states on the
   snapshot-only path with no paid-report body.
4. Add focused Node tests for locked, unlocked, and hostile artifact Markdown.
5. Leave ATLAS proxy routes, Stripe Checkout, and browser SPA behavior
   unchanged.

### Files touched

- `plans/PR-FAQ-Deflection-Portfolio-Paid-Report-Renderer.md`
- `portfolio-ui/api/content-ops/deflection/result-page.js`
- `portfolio-ui/scripts/faq-deflection-atlas-proxy.test.mjs`

## Mechanism

`renderResultPage(...)` continues to receive the sanitized server-side proxy
envelope. A new paid-artifact renderer checks all of these conditions before
emitting the full report section:

```text
report.ok === true
report.artifact_status === "unlocked"
typeof report.artifact.markdown === "string"
report.artifact.markdown.trim() is non-empty
```

When the conditions pass, the Markdown is escaped and placed inside a `<pre>`
element marked with `data-atlas-deflection-paid-report`. When they fail, no paid
report marker or artifact body is rendered. The unlock panel switches to an
unlocked state after ATLAS reports the artifact unlocked, which avoids offering
a second Checkout session for an already unlocked report.

## Intentional

- This slice does not add a Markdown parser. Rendering artifact Markdown as
  escaped text proves the paid unlock path without introducing an HTML
  sanitization surface.
- The renderer trusts only the ATLAS artifact envelope, not the snapshot. Free
  snapshot fields cannot create paid report copy.
- Missing or malformed artifact Markdown stays out of the DOM even when the
  envelope says `unlocked`; a future rich renderer can add more structured
  artifact validation.
- The browser SPA remains a validation shell. The hosted server-rendered page is
  the product path that has access to the server-only proxy.

## Deferred

- Rich Markdown styling, section navigation, and downloadable report exports
  after live paid artifact validation proves the shape customers should see.
- Browser SPA paid-state polish after the server-rendered hosted page is
  verified in production.
- Parked hardening: none.

## Verification

- `npm run test:deflection-atlas-proxy --prefix portfolio-ui` - passed, 11 checks.
- `npm run test:deflection-result --prefix portfolio-ui` - passed, 11 checks.
- `npm run build --prefix portfolio-ui` - passed using the existing ignored root
  `portfolio-ui/node_modules` install for local verification; `npm ci` in this
  worktree is blocked by pre-existing portfolio lockfile drift.
- `bash scripts/run_extracted_pipeline_checks.sh` - extracted_reasoning_core
  295 passed; extracted_content_pipeline 2855 passed, 10 skipped, 1 warning.
- `bash scripts/local_pr_review.sh --current-pr-body-file /tmp/faq-deflection-paid-report-renderer-pr-body.md` -
  passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan doc | ~80 |
| Hosted paid renderer | ~80 |
| Tests | ~50 |
| **Total** | **~210** |

# PR-FAQ-Deflection-Portfolio-Result-Page-UI

## Why this slice exists

The FAQ deflection backend now returns a locked report snapshot and the
portfolio result-page validation harness is on `main`, but the portfolio app
does not yet have a customer-facing result route that emits the required
Checkout metadata or unlock controls. Without that route, the hosted validation
smoke has no real portfolio page to exercise and the Stripe handoff remains a
documented contract instead of a product surface.

## Scope (this PR)

Ownership lane: portfolio-ui/faq-deflection
Slice phase: Vertical slice

1. Add a hosted portfolio route for FAQ deflection report results keyed by
   `{request_id}` and `account_id`.
2. Render the free snapshot only when it is already available from the submit
   flow, and strip/fail closed on paid-report fields.
3. Add a server-side portfolio Checkout endpoint that creates Stripe Checkout
   sessions with the required deflection metadata.
4. Wire the result page unlock action to that server endpoint without exposing
   ATLAS service credentials in browser code.
5. Add focused portfolio tests for the route markers, trust boundary, and
   Checkout request contract.

### Files touched

- `plans/PR-FAQ-Deflection-Portfolio-Result-Page-UI.md`
- `.github/workflows/portfolio_ui_checks.yml`
- `portfolio-ui/package.json`
- `portfolio-ui/vercel.json`
- `portfolio-ui/src/App.tsx`
- `portfolio-ui/src/pages/FaqDeflectionResult.tsx`
- `portfolio-ui/api/content-ops/deflection/checkout.js`
- `portfolio-ui/api/content-ops/deflection/result-page.js`
- `portfolio-ui/scripts/faq-deflection-result-page.test.mjs`

## Mechanism

The hosted route `/services/faq-deflection/results/:requestId` rewrites to a
Vercel function that returns HTML containing the stable
`data-atlas-deflection-*` hooks, `request_id`, `account_id`, and
`content_ops_deflection_report` markers required by the hosted validation
harness. The SPA route uses the same URL locally and looks for the free snapshot
in `sessionStorage` under
`atlas:deflection:snapshot:{request_id}`. Snapshot rendering is limited to
`summary` and `top_questions`; any forbidden paid-report key such as `markdown`,
`faq_result`, `answer`, `evidence`, or `source_ids` causes the page to hide the
snapshot and show a review state instead.

The unlock button posts `{ request_id, account_id }` to
`/api/content-ops/deflection/checkout`. That Vercel function validates the
fields, builds a one-time Stripe Checkout Session with:

```json
{
  "mode": "payment",
  "metadata": {
    "source": "content_ops_deflection_report",
    "account_id": "...",
    "request_id": "..."
  }
}
```

and redirects the browser only to Stripe's returned Checkout URL. The function
does not call the privileged ATLAS `/paid` route; ATLAS still unlocks only from
the signed Stripe webhook.

## Intentional

- This slice does not place `ATLAS_B2B_JWT`, `ATLAS_API_BASE_URL`, or any
  service credential in Vite/browser code. The current portfolio app is static
  Vite plus Vercel functions, so customer-visible ATLAS reads need a separate
  server proxy before the browser can hydrate `/snapshot` or `/artifact`
  directly.
- The server-rendered HTML route validates `account_id` before rendering and
  escapes script-serialized values defensively so the hosted customer route
  cannot reflect script terminators from query strings.
- The page does not synthesize estimates when no snapshot is available. It
  displays only real snapshot numbers already provided by the submit flow.
- Checkout uses Stripe's direct API through `fetch` instead of adding the
  Stripe SDK dependency in this vertical slice.
- If operators configure `STRIPE_DEFLECTION_REPORT_PRICE_ID`, the checkout
  handler validates that Stripe Price object is active, USD, and at least
  150000 cents before creating a Checkout session. This prevents a dashboard
  price misconfiguration from taking payment that ATLAS would later refuse to
  unlock.
- The paid artifact view is not implemented here. The unlock success return
  route keeps the page visible while ATLAS waits for the verified Stripe
  webhook to mark the report paid.

## Deferred

- Add a secure portfolio server proxy for ATLAS snapshot/artifact hydration
  after the result route is live.
- Add the paid artifact renderer after the proxy can fetch
  `/content-ops/deflection-reports/{request_id}/artifact` server-side.
- Add private Vercel Blob upload/submit UX before the result route if the
  portfolio needs a fully self-serve upload funnel.
- Parked hardening: none.

## Verification

- `npm run test:deflection-result --prefix portfolio-ui` - passed, 11 checks.
- `npm run build --prefix portfolio-ui` - passed using the existing ignored
  root `portfolio-ui/node_modules` install for local verification; `npm ci` in
  this worktree is blocked by pre-existing portfolio lockfile drift.
- `bash scripts/local_pr_review.sh --current-pr-body-file /tmp/faq-deflection-result-page-ui-pr-body.md` -
  passed after the review fixes.
- `bash scripts/run_extracted_pipeline_checks.sh` - extracted_reasoning_core
  295 passed; extracted_content_pipeline 2845 passed, 10 skipped, 1 warning.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan doc | ~132 |
| Portfolio UI workflow | ~38 |
| Route wiring | ~12 |
| Result page | ~330 |
| Checkout API | ~215 |
| Hosted result HTML function | ~175 |
| Focused test script | ~265 |
| Package script | ~1 |
| **Total** | **~1168** |

The slice is over the 400 LOC target because the usable route, server-side
Checkout creation, hosted raw-HTML validation surface, paid-field guard, and
focused contract tests are the minimum vertical customer path that validates the
trust boundary without leaking credentials.

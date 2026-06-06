# PR-FAQ-Deflection-Portfolio-Atlas-Proxy

## Why this slice exists

PR-FAQ-Deflection-Portfolio-Result-Page-UI added the hosted result page and
server-side Checkout creation, but it intentionally did not let browser code
call ATLAS because that would expose the B2B service JWT. The next product step
is a portfolio-owned server proxy that can read ATLAS snapshot/artifact state
with secret server env vars and render real free snapshot numbers on the hosted
result page.

This keeps the paywall trust boundary intact: the portfolio can display the
free snapshot and read paid/unpaid artifact state, but ATLAS still releases the
full artifact only after Stripe's verified webhook marks the report paid.

## Scope (this PR)

Ownership lane: portfolio-ui/faq-deflection
Slice phase: Vertical slice

1. Add a server-only portfolio ATLAS proxy helper for deflection report
   snapshot/artifact hydration.
2. Add a public portfolio API route that validates `request_id` and
   `account_id`, checks that the account matches the configured portfolio
   service account, and returns a locked/unlocked report envelope.
3. Render real free snapshot metrics and top questions in the hosted result
   page when ATLAS returns a snapshot.
4. Keep full paid artifact rendering deferred; this slice only reports
   `artifact_status` and never derives paid copy from the snapshot.
5. Enroll the proxy tests in the existing portfolio UI CI lane.

### Files touched

- `plans/PR-FAQ-Deflection-Portfolio-Atlas-Proxy.md`
- `portfolio-ui/api/content-ops/deflection/atlas-report.js`
- `portfolio-ui/api/content-ops/deflection/report.js`
- `portfolio-ui/api/content-ops/deflection/result-page.js`
- `portfolio-ui/package.json`
- `portfolio-ui/scripts/faq-deflection-atlas-proxy.test.mjs`
- `portfolio-ui/scripts/faq-deflection-result-page.test.mjs`
- `.github/workflows/portfolio_ui_checks.yml`

## Mechanism

`atlas-report.js` owns the server-only ATLAS integration. It reads
`ATLAS_API_BASE_URL`, `ATLAS_B2B_JWT`, and `ATLAS_ACCOUNT_ID` from server env,
rejects missing config, rejects account mismatches, then calls:

```text
GET /api/v1/content-ops/deflection-reports/{request_id}/snapshot
GET /api/v1/content-ops/deflection-reports/{request_id}/artifact
```

with `Authorization: Bearer <ATLAS_B2B_JWT>` and per-fetch abort timeouts.
Snapshot responses are whitelist-projected into only the known free fields:
`summary.{generated,drafted_answer_count,no_proven_answer_count}` and
`top_questions[].{rank,question,weighted_frequency,customer_wording}`. Extra
fields from ATLAS are dropped before the browser-visible envelope is returned.
Artifact `403` becomes `artifact_status: "locked"`, `200` becomes
`"unlocked"`, and `404` becomes `"missing"`.

The hosted result page calls the helper server-side and renders only snapshot
summary/top-question fields. The browser-visible API route returns the same
sanitized envelope without the token.

## Intentional

- The proxy is bound to the configured `ATLAS_ACCOUNT_ID`. The public
  `account_id` query is validation and Checkout metadata continuity, not a
  tenant selector for arbitrary ATLAS accounts.
- The full paid artifact renderer remains out of scope. Returning
  `artifact_status` proves paid-state hydration without adding Markdown/FAQ
  rendering in this slice.
- The server result page still renders without a snapshot when ATLAS is
  unavailable or times out; it does not synthesize counts or fallback estimates.
- The route uses server env vars only; no `VITE_` env or browser code receives
  the ATLAS JWT.
- Public config failures omit env-var detail strings from the browser response.

## Deferred

- Render the paid `FAQDeflectionReportArtifact` after the proxy is deployed and
  live artifact hydration is verified.
- Add a customer-facing upload/submit form ahead of this result page once the
  private Blob upload path is chosen.
- Parked hardening: none.

## Verification

- `npm run test:deflection-result --prefix portfolio-ui` - passed, 11 checks.
- `npm run test:deflection-atlas-proxy --prefix portfolio-ui` - passed, 9 checks.
- `npm run build --prefix portfolio-ui` - passed using the existing ignored
  root `portfolio-ui/node_modules` install for local verification; `npm ci` in
  this worktree is blocked by pre-existing portfolio lockfile drift.
- `bash scripts/run_extracted_pipeline_checks.sh` - extracted_reasoning_core
  295 passed; extracted_content_pipeline 2853 passed, 10 skipped, 1 warning.
- `bash scripts/local_pr_review.sh --current-pr-body-file /tmp/faq-deflection-portfolio-atlas-proxy-pr-body.md` -
  passed after the review fixes.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan doc | ~113 |
| Proxy helper/API | ~370 |
| Hosted result page hydration | ~120 |
| Tests and package/workflow wiring | ~320 |
| **Total** | **~923** |

The slice is above 400 LOC because the secure proxy, fail-closed snapshot
sanitizer, hosted page hydration, API envelope, and CI-enrolled negative
fixtures are the minimum safe vertical path.

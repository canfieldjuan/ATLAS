# PR-FAQ-Deflection-Portfolio-Live-Submit

## Why this slice exists

PR-FAQ-Deflection-Portfolio-Upload-Shell added the customer-facing portfolio
upload route and a fail-closed submit endpoint. PR-FAQ-Deflection-Submit-
Multipart-CSV is now on `main`, so the portfolio can complete the next trust
boundary slice: accept the browser CSV form, forward raw multipart bytes to
ATLAS from the server, and return the canonical result path without exposing
ATLAS credentials to the browser.

This is the thinnest live submit path after the upload shell. It does not add
Blob persistence or rich progress tracking; it proves the server-to-server
handoff and keeps the result page responsible for hydrating real snapshot and
paid artifact state.

## Scope (this PR)

Ownership lane: portfolio-ui/faq-deflection
Slice phase: Vertical slice

1. Enable the portfolio upload page to submit `multipart/form-data` to the
   portfolio server endpoint.
2. Forward the raw multipart body from the portfolio server to ATLAS
   `/api/v1/content-ops/deflection-reports/submit` with the server-only B2B
   JWT.
3. Bind the submit response to the configured `ATLAS_ACCOUNT_ID`; reject browser
   account mismatches before calling ATLAS.
4. Return only `request_id`, `account_id`, and `result_path` to the browser,
   then redirect to the hosted result page.
5. Keep paid artifact rendering and snapshot hydration owned by the existing
   result page/proxy path.

### Files touched

- `plans/PR-FAQ-Deflection-Portfolio-Live-Submit.md`
- `portfolio-ui/api/content-ops/deflection/submit.js`
- `portfolio-ui/src/pages/FaqDeflectionUpload.tsx`
- `portfolio-ui/scripts/faq-deflection-upload-shell.test.mjs`

## Mechanism

The browser page builds a `FormData` body with `csv_file`, `support_platform`,
`company_name`, `contact_email`, and `limit`. It sends the user-entered
`account_id` in an `X-Atlas-Account-Id` header only for server-side binding
validation; ATLAS still derives account scope from the service JWT.

The portfolio endpoint validates:

```text
method === POST
Content-Type includes multipart/form-data
ATLAS_API_BASE_URL, ATLAS_B2B_JWT, and ATLAS_ACCOUNT_ID are configured
X-Atlas-Account-Id matches ATLAS_ACCOUNT_ID
Content-Length/body stay below the Vercel Functions request cap for this live path
```

It then forwards the raw multipart bytes to ATLAS with
`Authorization: Bearer <ATLAS_B2B_JWT>` and the configured timeout as an
`AbortController` signal. On ATLAS success, the response is projected to
`{ ok, request_id, account_id, result_path }`; the execute envelope is not
returned to the browser. The browser redirects to `result_path`, where the
existing server-rendered result page hydrates the real snapshot and paid artifact
state.

## Intentional

- The portfolio endpoint does not parse CSV rows or synthesize snapshot counts.
  ATLAS remains the source of truth for report creation and result hydration.
- The public response does not include the ATLAS execute envelope, support
  ticket rows, JWT, or env-var names.
- The browser supplies `account_id` for continuity only; the server enforces
  equality with the configured service account before forwarding.
- The live portfolio path caps selected CSVs at 4 MB so Vercel can deliver the
  request to the API route; the backend's larger submit limit remains available
  for a later direct/Blob upload flow.
- Freshdesk is submitted as `other`, and Help Scout as `help_scout`, matching
  the ATLAS submit validator's accepted support-platform values.
- Private Blob persistence remains deferred. This slice submits the browser
  selected CSV directly through the portfolio server.

## Deferred

- Add private Vercel Blob persistence if uploads need durable storage before
  forwarding bytes to ATLAS or need to preserve the backend's larger 50 MB CSV
  submit limit from the browser.
- Add richer progress/retry UX after the live submit path has production
  verification.
- Parked hardening: none.

## Verification

- Command: npm run test:deflection-upload-shell --prefix portfolio-ui
  - Result: passed, 10 checks.
- Command: npm run test:deflection-result --prefix portfolio-ui
  - Result: passed, 11 checks.
- Command: npm run test:deflection-atlas-proxy --prefix portfolio-ui
  - Result: passed, 11 checks.
- Command: npm run build --prefix portfolio-ui
  - Result: passed using the existing ignored
  root `portfolio-ui/node_modules` install for local verification; `npm ci` in
  this worktree is blocked by pre-existing portfolio lockfile drift.
- Command: bash scripts/run_extracted_pipeline_checks.sh
  - Result: extracted_reasoning_core 295 passed; extracted_content_pipeline
    2859 passed, 10 skipped, 1 warning.
- Command: bash scripts/local_pr_review.sh --current-pr-body-file /home/juan-canfield/Desktop/atlas-pr-bodies/faq-deflection-portfolio-live-submit.md
  - Result: passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan doc | ~115 |
| Submit endpoint forwarding | ~185 |
| Upload page live submit | ~90 |
| Tests | ~215 |
| **Total** | **~605** |

This slice is above 400 LOC because the trust-boundary forwarding endpoint,
browser submit state, and negative fixtures need to ship together. Splitting
the tests away from the endpoint would leave the credential boundary
under-proven.

# PR-FAQ-Deflection-Portfolio-Upload-Shell

## Why this slice exists

The FAQ deflection portfolio result page can now show the free snapshot, create
Checkout, hydrate ATLAS state through a server proxy, and render the paid
artifact after unlock. The missing portfolio-owned entry point is the submit
page before that result route.

The backend multipart submit contract has now landed on `main`, but wiring the
portfolio server-to-ATLAS upload is a separate trust-boundary slice. This PR
adds the customer-facing portfolio upload shell and a fail-closed portfolio
submit endpoint so the next slice can replace the guarded response with the
server-side ATLAS handoff without also introducing the page.

## Scope (this PR)

Ownership lane: portfolio-ui/faq-deflection
Slice phase: Vertical slice

1. Add a portfolio FAQ deflection upload route at `/services/faq-deflection`.
2. Render a CSV upload shell with company/contact/platform/account fields and
   client-side file readiness state.
3. Add a portfolio server submit endpoint that rejects requests with a
   non-2xx guard response until the live ATLAS forwarding slice lands.
4. Link the upload route into the services page and keep the result route
   unchanged.
5. Add focused portfolio tests and enroll them in the existing portfolio UI CI
   workflow.

### Files touched

- `plans/PR-FAQ-Deflection-Portfolio-Upload-Shell.md`
- `.github/workflows/portfolio_ui_checks.yml`
- `portfolio-ui/package.json`
- `portfolio-ui/src/App.tsx`
- `portfolio-ui/src/pages/Services.tsx`
- `portfolio-ui/src/pages/FaqDeflectionUpload.tsx`
- `portfolio-ui/api/content-ops/deflection/submit.js`
- `portfolio-ui/scripts/faq-deflection-upload-shell.test.mjs`

## Mechanism

The new React route is a portfolio-owned shell for preparing the CSV handoff.
It keeps all entered data in local component state, validates that the selected
file looks like a CSV and is within the 50 MB portfolio cap, and marks the
submit action disabled until the server contract is wired. No browser code
contains ATLAS host or JWT configuration.

The new `/api/content-ops/deflection/submit` handler is intentionally
fail-closed. It accepts only `POST`, sets `Cache-Control: no-store`, and returns
`503 { ok: false, error: "deflection_submit_backend_pending" }`. It does not
call ATLAS or read service credentials. This gives the portfolio a stable route
and test marker without pretending the portfolio server-to-ATLAS forwarding
path is live before that trust-boundary slice is built.

## Intentional

- The shell does not upload to ATLAS yet. This keeps the customer-facing route
  and the server credential forwarding path in separate reviewable slices.
- The server endpoint returns non-2xx while live forwarding is pending so no
  user can mistake the shell for a completed report generation path.
- The upload page uses existing portfolio styling and route-level lazy loading
  instead of adding a design-system dependency.
- The result page remains the only route that can unlock or render the paid
  artifact.

## Deferred

- Wire the server submit endpoint to the ATLAS multipart submit contract now
  that the backend contract is on `main`.
- Add private Vercel Blob persistence if the portfolio needs to store uploads
  before forwarding bytes to ATLAS.
- Parked hardening: none.

## Verification

- Command: npm run test:deflection-upload-shell --prefix portfolio-ui
  - Result: passed, 6 checks.
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
- Command: bash scripts/local_pr_review.sh --current-pr-body-file /home/juan-canfield/Desktop/atlas-pr-bodies/faq-deflection-portfolio-upload-shell.md
  - Result: passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan doc | ~90 |
| Upload shell route/page | ~220 |
| Guarded submit endpoint | ~45 |
| Tests/package/workflow | ~120 |
| Services link | ~20 |
| **Total** | **~495** |

This slice is above 400 LOC because it adds a new customer-facing route, the
fail-closed server boundary, and CI-enrolled tests together. Splitting the page
from the guarded endpoint would leave the shell without the trust-boundary
regression checks reviewers need.

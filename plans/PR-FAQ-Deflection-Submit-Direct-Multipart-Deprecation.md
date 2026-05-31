# PR-FAQ-Deflection-Submit-Direct-Multipart-Deprecation

## Why this slice exists

The FAQ deflection upload product path now sends the selected CSV to private
Vercel Blob, then submits a server-side JSON Blob reference to the portfolio
submit route. The older browser-to-portfolio raw multipart path remains in
`portfolio-ui/api/content-ops/deflection/submit.js`, but no current public UI
uses it.

Keeping that compatibility path preserves extra body-reading and account-header
surface area after the private-Blob path has become the real intake flow. This
slice deprecates direct multipart at the portfolio edge so uploads use the
configured server account and private Blob handoff only.

## Scope (this PR)

Ownership lane: content-ops/deflection-report-gating
Slice phase: Production hardening

1. Reject direct `multipart/form-data` requests to the portfolio submit route
   with an explicit deprecated-path response before config or ATLAS access.
2. Keep JSON private-Blob submit behavior unchanged.
3. Remove raw multipart body-reading helpers and exports that are no longer
   reachable.
4. Update focused portfolio upload-shell tests to prove the UI uses private
   Blob JSON submit and direct multipart cannot reach ATLAS.

### Files touched

- `plans/PR-FAQ-Deflection-Submit-Direct-Multipart-Deprecation.md`
- `portfolio-ui/api/content-ops/deflection/submit.js`
- `portfolio-ui/scripts/faq-deflection-upload-shell.test.mjs`

## Mechanism

The route now accepts only JSON private-Blob submits after the `POST` method
check. If `Content-Type` includes `multipart/form-data`, it returns `410` with
`direct_multipart_deprecated` and exits before loading ATLAS config, checking
account headers, reading the request body, or calling `fetch()`.

Non-JSON/non-multipart requests still return `415`, now with a JSON-specific
error. The private-Blob path continues to read the configured server account,
load the private CSV object, reconstruct ATLAS `FormData`, and call
`forwardSubmit()`.

## Intentional

- This deprecates only the portfolio edge's direct multipart compatibility
  path. ATLAS still accepts multipart because the portfolio server reconstructs
  multipart from private Blob for the real flow.
- The submit live smoke's `--csv-file` local fixture path remains useful for
  direct helper validation; route-handler mode remains private-Blob JSON only.
- This keeps matching legacy `X-Atlas-Account-Id` headers tolerated on JSON
  submits for now, while mismatches still fail closed.

## Deferred

- Removing the submit smoke's local CSV fixture mode is deferred because it
  validates `submitPrivateBlob()` without needing a live Blob object.
- Parked hardening: none.

## Verification

- Node syntax check for the submit route and focused upload-shell test - passed.
- `npm run test:deflection-upload-shell --prefix portfolio-ui` - 20 checks passed.
- `npm run build --prefix portfolio-ui` - passed after hydrating local
  `portfolio-ui/node_modules` with `npm install --prefix portfolio-ui`; Vite
  emitted the existing large-chunk advisory and skipped sitemap generation
  because `PORTFOLIO_SITE_URL` / `VITE_SITE_URL` is not set.
- Local PR review with the prepared PR body file - passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan doc | 90 |
| Submit route deprecation and helper removal | 120 |
| Focused tests | 95 |
| **Total** | **305** |

Under the 400 LOC soft cap.

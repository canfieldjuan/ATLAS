# PR-Deflection-Inspect-Preview-Gate

## Why this slice exists

#1384's remaining ATLAS-side launch gap is the pre-payment inspect gate. CSV
parser hardening, HTML normalization, and provider-shaped fixtures are merged,
but the public FAQ deflection upload still goes straight from browser -> private
Blob -> submit/report generation. A bad or ticket-index-only CSV can therefore
reach the locked-report path before the buyer sees the existing ingestion
diagnostics.

This slice gates the public portfolio submit flow with the existing ATLAS
`/content-ops/ingestion/files/inspect` contract before private Blob upload and
report generation. It does not invent a new parser or LLM path.

The diff is over the 400 LOC soft cap because the gate is indivisible at this
layer: the server proxy, browser block, submit-route enforcement, and
CI-facing negative tests must land together or the public page either exposes an
unauthenticated client call, shows diagnostics without enforcement, leaves the
submit API bypassable, or enforces a gate with no failure-branch coverage.

## Scope (this PR)

Ownership lane: content-ops/deflection-launch-readiness
Slice phase: Functional validation

1. Add a portfolio server route that forwards browser multipart inspect requests
   to ATLAS with server-side auth, byte limits, and fail-closed response
   projection.
2. Run inspect from `FaqDeflectionUpload` before private Blob upload; block
   normal report creation when inspect fails or returns `ok: false`.
3. Re-inspect the private Blob server-side in the submit route before forwarding
   to ATLAS report generation so direct submit API calls cannot bypass the gate.
4. Show bounded diagnostics in the upload page: ticket/source count, source
   type counts, warning/missing-field counts, and sample preview.
5. Extend the enrolled portfolio upload shell test for success and failure
   paths. No new test script or workflow enrollment is needed.

### Review Contract

- Acceptance criteria:
  - [ ] The browser never receives ATLAS URL/token/account credentials.
  - [ ] The inspect route forwards multipart bytes to ATLAS with auth and caps
        upload size before forwarding.
  - [ ] Malformed ATLAS inspect envelopes fail closed.
  - [ ] `ok: false` inspect diagnostics block browser Blob upload and direct
        private-Blob report submit.
  - [ ] `ok: true` diagnostics allow the existing private Blob submit path.
  - [ ] UI copy calls this a pre-flight/preview, not a paid-report result.
- Affected surfaces: portfolio deflection API route, public upload page, enrolled
  portfolio upload shell test.
- Risk areas: secret leakage, double-upload ergonomics, malformed upstream
  envelope handling, false-green source tests.
- Reviewer rules triggered: R1, R2, R3, R5, R9, R10, R12, R13.

### Files touched

- `plans/PR-Deflection-Inspect-Preview-Gate.md`
- `portfolio-ui/api/content-ops/deflection/inspect.js`
- `portfolio-ui/api/content-ops/deflection/submit.js`
- `portfolio-ui/scripts/faq-deflection-upload-shell.test.mjs`
- `portfolio-ui/src/pages/FaqDeflectionUpload.tsx`

## Mechanism

`FaqDeflectionUpload` builds a `FormData` payload with the selected CSV and
deflection defaults (`source_rows=true`, `source=ticket-csv-upload`,
`target_mode=faq_deflection_report`, `file_format=csv`, small sample limit).
It posts that payload to `/api/content-ops/deflection/inspect`.

The new portfolio route disables body parsing, reads the raw multipart request
with a strict byte cap, and forwards the same body to ATLAS
`/api/v1/content-ops/ingestion/files/inspect` using `ATLAS_B2B_JWT` from the
server environment. It projects only bounded diagnostics back to the browser.

If browser inspect fails, is malformed, or returns `ok: false`, the UI renders
the diagnostics/error and does not start private Blob upload. If browser inspect
succeeds, the private Blob upload runs.

The submit route does not trust that browser state. After reading the private
Blob, it builds its own ATLAS inspect `FormData` from the stored CSV, requires
`ok: true`, and only then forwards to `/api/v1/content-ops/deflection-reports/submit`.
A direct API caller with only a `blob_pathname` therefore cannot bypass the
pre-payment gate.

## Intentional

- This is a validation gate, not a replacement for the paid result page.
- The normal UI path inspects before Blob upload, then the submit route
  re-inspects the private Blob. That duplicate deterministic inspect is
  intentional because the submit API must not trust React state.
- The route projects diagnostics instead of returning the full upstream payload
  so source material and raw emails do not leak into the browser by accident.

## Deferred

- Richer provider-specific remediation copy can be added after this gate lands.
- True operator-supplied sanitized provider exports remain a stronger #1384 proof
  when supplied.

Parked hardening: none.

## Verification

- `npm ci` (portfolio-ui dependencies installed for local build)
- `npm run build` (passes; Vite reports existing sitemap/chunk-size warnings)
- `npm run test:deflection-upload-shell` (25/25 ok)
- `npm run test:deflection-result` (18/18 ok)
- `npm run test:deflection-atlas-proxy` (17/17 ok)
- `git diff --check`

## Estimated diff size

| File | LOC |
|---|---:|
| `plans/PR-Deflection-Inspect-Preview-Gate.md` | 121 |
| `portfolio-ui/api/content-ops/deflection/inspect.js` | 224 |
| `portfolio-ui/api/content-ops/deflection/submit.js` | 39 |
| `portfolio-ui/scripts/faq-deflection-upload-shell.test.mjs` | 365 |
| `portfolio-ui/src/pages/FaqDeflectionUpload.tsx` | 214 |
| **Total** | **963** |

# PR-Deflection-Zendesk-Credential-Flow

## Why this slice exists

#1527 added the hosted, tenant-scoped Zendesk full-thread export route and
explicitly deferred the portfolio UI flow that starts from stored Zendesk
credentials. The product can already upload a saved Zendesk JSON export through
private Blob, but a buyer with configured Zendesk credentials still has to leave
the intake page, export by hand, and upload the JSON file. This slice closes the
next vertical handoff: the portfolio intake can start a server-side Zendesk API
export, persist the returned artifact privately, and reuse the existing
full-thread submit path without exposing Zendesk API tokens, ATLAS service
tokens, private Blob tokens, or the exported ticket artifact to browser-visible
payloads. The first review also caught that the route itself cannot stay
public: this slice now gates the stored-credential export with a narrow
operator-entered route access token before any ATLAS/Zendesk/Blob work starts.
This is over the 400 LOC soft cap because the slice is intentionally vertical
across the portfolio server proxy, the no-file UI mode, and the enrolled shell
tests that prove browser secrecy, fail-closed export handling, private Blob
persistence, and submit reuse in one reviewable contract.

## Scope (this PR)

Ownership lane: content-ops/deflection-launch-readiness
Slice phase: Vertical slice

1. Add a portfolio server proxy that calls the #1527 ATLAS hosted Zendesk
   export route with the server-side ATLAS token, validates the full-thread
   artifact envelope, writes the artifact to private Vercel Blob, then reuses
   the existing `submitPrivateBlob` full-thread path.
2. Add a "Zendesk API" intake mode to the portfolio upload page. CSV and saved
   Zendesk JSON upload modes remain unchanged. The API mode requires a route
   access token entered by the operator; it is not a public unauthenticated
   export trigger.
3. Add CI-facing tests for proxy request shape, fail-closed export/Blob/submit
   errors, sanitized responses, UI markers, and unchanged CSV/JSON upload
   behavior through the existing enrolled shell test.

### Review Contract
- Acceptance criteria:
  - [ ] CSV remains the default upload mode and still requires CSV inspection
        before private Blob upload.
  - [ ] Saved Zendesk JSON upload mode remains unchanged and still submits
        `importer_mode="full_thread"` through private Blob.
  - [ ] Zendesk API mode does not require a browser file upload and calls only
        a portfolio server endpoint after the operator provides the route
        access token; the browser never receives ATLAS tokens, Zendesk
        credentials, Blob tokens, private Blob URLs, or the exported artifact.
  - [ ] The server proxy rejects missing or incorrect route access tokens before
        any ATLAS export, Blob write, or submit attempt.
  - [ ] The server proxy calls `/api/v1/content-ops/zendesk-export/full-thread`
        with the server-side bearer token and no caller-supplied account id.
  - [ ] Export failures, malformed export envelopes, Blob write failures, and
        submit failures fail closed with static public error codes and no token
        or raw upstream body leakage.
  - [ ] The exported artifact is written as private JSON Blob and then submitted
        via the existing full-thread `submitPrivateBlob` helper so importer and
        cleanup behavior stay centralized.
- Affected surfaces: portfolio FAQ deflection intake page, portfolio
  deflection API proxy, enrolled portfolio shell test.
- Risk areas: browser-visible ticket data, cross-tenant account spoofing,
  private Blob path confusion, duplicate submit helper logic, stale frontend CI
  enrollment.
- Reviewer rules triggered: R1, R2, R3, R5, R9, R10, R12, R14.

### Files touched

- `plans/PR-Deflection-Zendesk-Credential-Flow.md`
- `portfolio-ui/api/content-ops/deflection/zendesk-export-submit.js`
- `portfolio-ui/scripts/faq-deflection-upload-shell.test.mjs`
- `portfolio-ui/src/pages/FaqDeflectionUpload.tsx`

## Mechanism

The upload page gets a third `uploadMode`, `zendesk_api`. In that mode the file
input is replaced with a credential-backed source panel plus a route access
token password field, and the submit button calls a new portfolio endpoint
instead of `uploadBlob`. Company/contact fields remain required; support
platform is pinned to Zendesk.

The new portfolio endpoint reads a small JSON request (`company_name`,
`contact_email`, `limit`, optional `start_time`). It builds config only from
server env via the existing `atlas-report.js` helpers and requires
`Authorization: Bearer <ATLAS_DEFLECTION_ZENDESK_EXPORT_ACCESS_TOKEN>` before
checking account headers or invoking ATLAS. After that route gate passes, it
calls the ATLAS export route with `Authorization: Bearer <server token>`,
validates that the response has `importer_mode="full_thread"`,
`support_platform="zendesk"`, and an artifact with a `tickets` array, then
writes `artifact` to private Vercel Blob as JSON.
After the private write succeeds, it calls the existing `submitPrivateBlob`
helper with `importer_mode="full_thread"` and the returned pathname. That helper
continues to own Blob readback, ATLAS multipart submit, and best-effort cleanup.

Public errors from this endpoint are static codes such as
`zendesk_export_unavailable`, `zendesk_export_contract_violation`,
`zendesk_export_blob_unavailable`, or the existing submit error. The endpoint
does not echo upstream response bodies, exception text, Authorization headers,
Blob tokens, or Zendesk credentials.

## Intentional

- No new backend route in this PR. #1527 already shipped the tenant-scoped
  ATLAS export route; this slice wires the portfolio product surface to it.
- No browser-visible exported artifact. The existing saved-JSON upload path is
  still available for manual exports, but credential-backed export keeps the
  fetched ticket artifact server-side.
- Browser-visible route access token, not service credentials. The operator
  supplies a narrow gate token for this public page; the server-side ATLAS
  token, Zendesk tenant credentials, and private Blob token are never embedded
  in the page or response.
- No new npm script. The existing enrolled `test:deflection-upload-shell`
  should cover the new UI/proxy branches to avoid another frontend CI enrollment
  gap.

## Deferred

- Browser-automated live smoke against the operator's Zendesk trial credentials
  after this endpoint is deployed.
- Optional export progress/polling UX if live Zendesk exports are too slow for
  a single request.
- Production provisioning of `ATLAS_DEFLECTION_ZENDESK_EXPORT_ACCESS_TOKEN`
  alongside the existing portfolio server env.

Parked hardening: none.

## Verification

- `cd portfolio-ui && npm run test:deflection-upload-shell` -- 37 passed.
- `cd portfolio-ui && npm run test:deflection-result` -- 19 passed.
- `cd portfolio-ui && npm run test:deflection-atlas-proxy` -- 18 passed.
- `cd portfolio-ui && npm run build` -- passed; Vite emitted the existing
  sitemap-env and large-chunk warnings.
- `scripts/run_extracted_pipeline_checks.sh` via bash -- 4063 passed, 10 skipped,
  1 existing torch/pynvml warning.

## Estimated diff size

| File | LOC |
|---|---:|
| `plans/PR-Deflection-Zendesk-Credential-Flow.md` | 144 |
| `portfolio-ui/api/content-ops/deflection/zendesk-export-submit.js` | 354 |
| `portfolio-ui/scripts/faq-deflection-upload-shell.test.mjs` | 523 |
| `portfolio-ui/src/pages/FaqDeflectionUpload.tsx` | 247 |
| **Total** | **1268** |

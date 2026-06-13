# PR-Deflection-Zendesk-Upload-Handoff

## Why this slice exists

PR #1520 proved the Zendesk `{ticket, comments}` full-thread importer and
backend `importer_mode="full_thread"` contract, but deliberately left the
browser upload UX CSV-only. That means a real Zendesk trial export still cannot
be submitted from the portfolio intake page without hand-building a backend
request.

This slice closes that handoff: the portfolio upload page can accept either the
existing support-ticket CSV or a Zendesk full-thread JSON export, store it in
private Vercel Blob, and submit it through the authenticated portfolio proxy to
ATLAS. The proxy keeps Blob private and forwards JSON bytes server-side; it does
not expose ATLAS tokens, Blob tokens, or private Blob URLs to the browser.

This may exceed the 400 LOC soft cap because the product path spans the
portfolio upload UI, the portfolio submit/upload proxies, the ATLAS multipart
submit parser, and CI-facing tests on both sides. Splitting the backend parser
from the UI would leave the browser with a visible JSON mode that cannot
complete a real report.

## Scope (this PR)

Ownership lane: content-ops/deflection-launch-readiness
Slice phase: Vertical slice

1. Add a portfolio upload mode selector for CSV vs Zendesk full-thread JSON.
   CSV remains the default and keeps CSV inspection before private Blob upload.
2. Allow private Blob uploads under the existing `faq-deflection/uploads/`
   prefix for JSON Zendesk exports as well as CSV.
3. Extend the portfolio submit proxy so JSON full-thread blobs are read
   server-side and forwarded to ATLAS as authenticated multipart `json_file`
   with `importer_mode="full_thread"`. CSV submit behavior remains unchanged.
4. Extend the ATLAS deflection submit multipart parser to accept
   `json_file` only when `importer_mode="full_thread"`; CSV mode still requires
   `csv_file`.
5. Add CI-facing tests for CSV unchanged behavior, JSON handoff request shape,
   fail-closed blob/path/mode validation, and ATLAS multipart full-thread
   parsing.

### Review Contract
- Acceptance criteria:
  - [ ] Omitted/CSV mode remains the default: CSV file validation, CSV inspect,
        private Blob upload, proxy cleanup, and ATLAS multipart `csv_file`
        submit still work.
  - [ ] Zendesk JSON mode accepts JSON files, uploads them privately with an
        allowed JSON content type, skips CSV inspect, and submits
        `importer_mode="full_thread"` to the portfolio proxy.
  - [ ] The portfolio proxy never exposes ATLAS credentials, Blob tokens, buyer
        account IDs, or private Blob URLs in browser-visible payloads.
  - [ ] JSON full-thread blobs are forwarded to ATLAS as authenticated
        multipart `json_file` with support metadata and `limit`.
  - [ ] Mismatched mode/path combinations fail closed: CSV mode cannot submit a
        JSON blob, full-thread mode cannot submit a CSV blob, unsafe paths are
        rejected before Blob/ATLAS calls, and malformed ATLAS envelopes still
        fail closed.
  - [ ] ATLAS multipart full-thread parsing produces the same diagnostics as
        the #1520 blob-url importer path and rejects missing `json_file`.
- Affected surfaces: portfolio upload page, portfolio Blob upload token route,
  portfolio submit proxy, ATLAS deflection submit API, portfolio and extracted
  tests.
- Risk areas: private data leakage, mode/path confusion, skipped CSV preflight,
  backend contract drift, frontend CI enrollment.
- Reviewer rules triggered: R1, R2, R3, R5, R9, R10, R12, R14.

### Files touched

- `extracted_content_pipeline/api/control_surfaces.py`
- `plans/PR-Deflection-Zendesk-Upload-Handoff.md`
- `portfolio-ui/api/content-ops/deflection/submit.js`
- `portfolio-ui/api/content-ops/deflection/upload.js`
- `portfolio-ui/scripts/faq-deflection-upload-shell.test.mjs`
- `portfolio-ui/src/pages/FaqDeflectionUpload.tsx`
- `tests/test_extracted_content_deflection_submit.py`

## Mechanism

The upload page gains an `uploadMode` state with two values:
`csv` and `zendesk_full_thread`. `fileState` validates the selected file against
that mode, and `blobPathname` preserves the corresponding CSV or JSON
suffix under the same private Blob prefix. CSV mode keeps the existing
inspect-before-upload path. Zendesk JSON mode skips the CSV inspect route,
uploads the JSON privately, then includes `importer_mode: "full_thread"` in the
portfolio submit payload.

The Blob token route admits JSON pathnames/content types in addition to the
existing CSV set, still scoped to `faq-deflection/uploads/`, still bound to the
server-configured account, and still capped at 50 MB.

The portfolio submit proxy replaces CSV-only helpers with mode-aware private
Blob helpers. CSV mode continues to read the blob, inspect it through ATLAS, and
forward multipart `csv_file`. Full-thread mode reads a JSON private blob,
skips CSV inspect, and forwards multipart `json_file` plus
`importer_mode="full_thread"` to ATLAS. Cleanup remains best-effort after the
side-effectful submit and must not mask a successful report creation.

The ATLAS submit parser accepts multipart `json_file` only when
`importer_mode="full_thread"` and parses it through the same
`load_zendesk_full_thread_rows_from_json_bytes` path used by #1520's blob-url
importer. CSV mode keeps requiring `csv_file`.

## Intentional

- No direct Zendesk API fetch in this slice; this is still an uploaded-export
  path. API export/import remains separate credentialed work.
- No CSV-style preflight inspect for full-thread JSON in the browser. The
  existing inspect route is a flat-file CSV ingestion preflight; JSON validation
  happens in ATLAS submit parsing after the private upload, with cleanup on
  failure.
- The portfolio does not pass private Blob URLs to ATLAS. The proxy reads
  private Blob bytes server-side and forwards authenticated multipart to avoid
  exposing Blob access.

## Deferred

- Direct Zendesk API export/import using tenant-scoped credentials.
- A dedicated JSON preflight diagnostics route if we need preview counts before
  private upload for full-thread exports.
- Live browser smoke against the operator's real Zendesk trial export.

Parked hardening: none.

## Verification

- `python -m pytest tests/test_extracted_content_deflection_submit.py -q`
  - 51 passed.
- `cd portfolio-ui && npm run test:deflection-upload-shell`
  - 27 passed.
- `cd portfolio-ui && npm run test:deflection-atlas-proxy`
  - 18 passed.
- `cd portfolio-ui && npm run build`
  - Passed after `npm ci` installed this worktree's local dependencies. Vite
    emitted the existing large-chunk and sitemap-env warnings.
- `scripts/run_extracted_pipeline_checks.sh` (via bash)
  - 4022 passed, 10 skipped, 1 existing torch CUDA warning.

## Estimated diff size

| File | LOC |
|---|---:|
| `extracted_content_pipeline/api/control_surfaces.py` | 52 |
| `plans/PR-Deflection-Zendesk-Upload-Handoff.md` | 149 |
| `portfolio-ui/api/content-ops/deflection/submit.js` | 110 |
| `portfolio-ui/api/content-ops/deflection/upload.js` | 16 |
| `portfolio-ui/scripts/faq-deflection-upload-shell.test.mjs` | 182 |
| `portfolio-ui/src/pages/FaqDeflectionUpload.tsx` | 162 |
| `tests/test_extracted_content_deflection_submit.py` | 73 |
| **Total** | **744** |

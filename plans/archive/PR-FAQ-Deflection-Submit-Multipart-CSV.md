# PR-FAQ-Deflection-Submit-Multipart-CSV

## Why this slice exists

Issue #1161 surfaced a production PII problem with the current
portfolio-to-ATLAS submit seam: private Vercel Blob objects do not provide a
native signed fetch URL, so keeping the `blob_url` contract would force the
portfolio to build a new public signed-proxy route for raw support-ticket CSVs.

This slice adds the safer production shape: portfolio reads its own private blob
server-side, then POSTs the CSV bytes to ATLAS over the existing authenticated
Bearer submit call.

The review pass found one production-critical parser edge: Starlette's default
multipart part limit can reject files above 1 MiB before the route's own 50 MB
bounded read runs. This PR now sets the parser part limit to the same CSV cap
and adds an early `Content-Length` guard before parsing the multipart body,
which pushes the final diff over the nominal 400 LOC budget.

## Scope (this PR)

Ownership lane: content-ops/deflection-report-gating
Slice phase: Production hardening

1. Extend `POST /content-ops/deflection-reports/submit` to accept
   `multipart/form-data` with `csv_file`, `support_platform`, `company_name`,
   `contact_email`, and optional `limit`.
2. Keep the existing JSON `blob_url` submit contract for compatibility while
   making multipart the documented production path.
3. Reuse the existing CSV normalization, 50 MB cap, 1000-row limit, truncation
   diagnostics, and synchronous execute envelope.
4. Match Starlette's multipart parser part limit to the advertised CSV cap and
   reject clearly oversize multipart bodies before form parsing.
5. Update frontend contract docs to steer production wiring away from a public
   signed proxy.

### Files touched

- `plans/PR-FAQ-Deflection-Submit-Multipart-CSV.md`
- `extracted_content_pipeline/api/control_surfaces.py`
- `tests/test_extracted_content_deflection_submit.py`
- `docs/frontend/content_ops_faq_deflection_checkout_contract.md`
- `tests/test_content_ops_faq_report_contract_docs.py`

## Mechanism

The submit route now inspects the request content type. Multipart requests read
`csv_file` with a bounded `max_bytes + 1` read and parse those bytes through the
same temp-file CSV loader used by the URL-fetching path. JSON requests continue
to validate `blob_url` and fetch through the existing DNS-pinned HTTPS loader.

Before reading multipart form data, the route rejects a `Content-Length` above
the CSV cap plus a small multipart overhead allowance. It then calls
`request.form(max_part_size=max_bytes)` so Starlette's parser accepts the same
file range the route advertises and the bounded upload reader enforces.

Both input shapes then feed the same support-ticket input package and
`faq_deflection_report` execution path. Diagnostics report `uploaded_bytes` for
multipart and keep `blob_bytes` for legacy JSON submits.

## Intentional

- The legacy `blob_url` path remains for compatibility and local rollback, but
  docs mark multipart as the production PII-safe path.
- This does not add browser upload or portfolio UI. Portfolio must still read
  its private Blob server-side and call ATLAS with the service JWT.
- The response shape remains the Content Ops execute envelope. No one-off
  flattened submit response is introduced.
- The `Content-Length` pre-check allows 1 MiB of multipart overhead above the
  CSV cap. That keeps normal boundary/header overhead from false-blocking while
  still bounding the disk-spool path before `request.form(...)`.

## Deferred

- Parked hardening: none.
- Removing the legacy `blob_url` path can be a later cleanup after portfolio
  production traffic has moved to multipart.

## Verification

- Command: python -m py_compile extracted_content_pipeline/api/control_surfaces.py tests/test_extracted_content_deflection_submit.py tests/test_content_ops_faq_report_contract_docs.py
  - Result: passed.
- Command: python -m pytest tests/test_extracted_content_deflection_submit.py tests/test_content_ops_faq_report_contract_docs.py -q
  - Result: 22 passed.
- Command: bash scripts/run_extracted_pipeline_checks.sh
  - Result: extracted_reasoning_core 295 passed; extracted_content_pipeline
    2859 passed, 10 skipped, 1 warning.
- Command: bash scripts/local_pr_review.sh --current-pr-body-file /home/juan-canfield/Desktop/atlas-pr-bodies/faq-deflection-submit-multipart-csv.md
  - Result: passed after review fix.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan doc | 104 |
| Submit route/helper | 196 |
| Tests | 157 |
| Docs | 34 |
| **Total** | **491** |

The slice exceeds 400 LOC because the review fix is part of the production
contract: without the explicit parser limit, the advertised 50 MB multipart path
can reject files above Starlette's default part cap before the route's own
bounded read runs.

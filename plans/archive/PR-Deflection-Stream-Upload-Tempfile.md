# PR-Deflection-Stream-Upload-Tempfile

## Why this slice exists

Issue #1458 flags the deflection submit multipart path as a launch-readiness
memory risk: the uploaded CSV is read into one full `bytes` object before being
written back to a temporary file for parsing. That creates an avoidable full-file
copy at the upload boundary, before the existing parser and row caps can help.
This vertical slice fixes the most-upstream safe boundary in scope: multipart CSV
uploads are copied directly to a bounded temp file in chunks, while the existing
CSV parser, diagnostics, and blob path remain unchanged.

## Scope (this PR)

Ownership lane: content-ops/deflection-launch-readiness
Slice phase: Vertical slice

1. Stream multipart CSV uploads for `/content-ops/deflection/submit` into a
   bounded temporary file instead of reading the whole upload into one `bytes`
   object.
2. Add upload-boundary regression coverage proving the helper uses chunked reads,
   preserves the current parser output, and still fails closed when the stream
   crosses the byte limit.

### Review Contract

- Acceptance criteria:
  - Multipart CSV submit no longer calls `read(max_bytes + 1)` on the upload
    object.
  - Oversize multipart CSV uploads still return the existing 413
    `deflection_submit_csv_too_large` envelope.
  - Existing CSV parsing behavior, load warnings, and byte-count diagnostics are
    preserved for successful uploads.
- Affected surfaces: `extracted_content_pipeline/api/control_surfaces.py`,
  `tests/test_extracted_content_deflection_submit.py`.
- Risk areas: temp-file cleanup, byte-count accounting, async upload fakes in
  tests, and preserving parser warning behavior.
- Reviewer rules triggered: R1, R2, R8, R9, R10, R13, R14.

### Files touched

- `extracted_content_pipeline/api/control_surfaces.py`
- `plans/PR-Deflection-Stream-Upload-Tempfile.md`
- `tests/test_extracted_content_deflection_submit.py`

## Mechanism

The deflection submit upload loader stops calling a full-limit upload read.
Instead it opens the existing submit temp file up front, then reads the
multipart upload in fixed-size chunks, writes each chunk to disk, and counts
bytes as it goes. If the count crosses `max_bytes`, it raises the same
`413 deflection_submit_csv_too_large` envelope as today. Empty uploads and
parser failures keep their existing behavior.

After the bounded copy succeeds, the helper parses that temp file through the
same `load_source_rows_with_warnings_from_file(..., file_format="csv")` path used
today, so delimiter/BOM/warning behavior stays centralized.

## Intentional

- This PR leaves the blob URL path unchanged. Blob fetches are the same root
  shape, but lower priority: they are bounded at 50MB and sourced from the
  app-controlled Vercel Blob path rather than direct browser multipart upload.
- This PR does not rewrite `load_source_rows_with_warnings_from_file()` into a
  streaming row iterator. The parser still returns a row list; this slice removes
  the avoidable full-upload `bytes` copy before parsing.
- The full-thread JSON upload path is unchanged because its shape is a separate
  importer mode and would need its own JSON streaming design.

## Deferred

- Follow-up hardening for #1458: stream rows out of the CSV parser instead of
  accumulating every parsed row before package construction, if live-volume
  profiling still shows memory pressure after this upload-boundary fix.
- Same-root deferred follow-up: stream the blob URL path into a bounded temp file
  instead of fetching the full blob into memory before parsing.
- Follow-up hardening: consider a streaming bounded JSON temp-file copy for
  `importer_mode=full_thread` if full-thread artifacts approach the submit byte
  limit in live trials.

Parked hardening: none.

## Verification

- python -m py_compile extracted_content_pipeline/api/control_surfaces.py tests/test_extracted_content_deflection_submit.py
- python -m pytest tests/test_extracted_content_deflection_submit.py -q (56 passed)
- bash scripts/check_ascii_python.sh
- bash scripts/validate_extracted_content_pipeline.sh
- python extracted/_shared/scripts/forbid_atlas_reasoning_imports.py extracted_content_pipeline
- python scripts/audit_extracted_standalone.py --fail-on-debt
- bash scripts/run_extracted_pipeline_checks.sh (reasoning core: 295 passed;
  content pipeline: 4185 passed, 10 skipped)
- python -m pytest tests/test_atlas_content_ops_generated_assets_api.py
  tests/test_content_ops_brand_voice_profiles.py
  tests/test_content_ops_brand_voice_profiles_api.py
  tests/test_content_ops_zendesk_credentials.py
  tests/test_content_ops_zendesk_export_api.py -q (62 passed)

## Estimated diff size

| File | LOC |
|---|---:|
| `extracted_content_pipeline/api/control_surfaces.py` | 121 |
| `plans/PR-Deflection-Stream-Upload-Tempfile.md` | 106 |
| `tests/test_extracted_content_deflection_submit.py` | 64 |
| **Total** | **291** |

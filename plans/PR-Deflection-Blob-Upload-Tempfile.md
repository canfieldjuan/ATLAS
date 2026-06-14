# PR-Deflection-Blob-Upload-Tempfile

## Why this slice exists

PR-Deflection-Stream-Upload-Tempfile closed the direct multipart upload memory
risk from #1458, but its review called out the same root still present in the
blob URL CSV path: `_read_bounded_https_blob(...)` returns one full `bytes`
object, then `_parse_deflection_submit_csv_bytes(...)` writes those bytes back
to a temp file for the CSV parser. That is the same avoidable full-file copy one
hop later in the submit flow. This slice drains that deferred same-root item for
CSV blobs by streaming the fetched response into the bounded temp file that the
parser already consumes.

## Scope (this PR)

Ownership lane: content-ops/deflection-launch-readiness
Slice phase: Production hardening

1. Stream `importer_mode=csv` blob URL responses into a bounded temp file
   instead of materializing the full blob as `bytes` before CSV parsing.
2. Preserve the existing blob URL security gates: HTTPS-only validation, DNS
   fail-closed checks, no redirects, non-2xx rejection, timeout, byte cap, and
   response close behavior.
3. Add regression coverage proving the CSV blob path reads in chunks, rejects
   oversize blobs before writing the over-limit chunk, closes responses, and
   still parses through the existing CSV loader.
4. Leave full-thread JSON blob parsing unchanged because
   `load_zendesk_full_thread_rows_from_json_bytes(...)` currently requires a
   byte payload and needs a separate JSON streaming design.

### Review Contract

- Acceptance criteria:
  - `importer_mode=csv` blob fetches no longer call `response.read(max_bytes + 1)`.
  - Oversize CSV blobs still return the existing 413
    `deflection_submit_blob_too_large` envelope.
  - CSV blob parsing still reports the same byte count and load-warning shape.
  - Redirect/non-success/malformed fetch handling and response cleanup remain
    fail-closed.
- Affected surfaces: `extracted_content_pipeline/api/control_surfaces.py`,
  `tests/test_extracted_content_deflection_submit.py`.
- Risk areas: response cleanup, partial temp-file cleanup, byte-count
  accounting, preserving full-thread JSON behavior, and broadening the SSRF
  fetch surface by accident.
- Reviewer rules triggered: R1, R2, R8, R9, R10, R13, R14.

### Files touched

- `extracted_content_pipeline/api/control_surfaces.py`
- `plans/PR-Deflection-Blob-Upload-Tempfile.md`
- `tests/test_extracted_content_deflection_submit.py`

## Mechanism

The blob loader keeps the existing async-to-thread boundary. Inside the sync
CSV path it opens the HTTP response through the same validated target helper,
checks redirect/status exactly as before, then copies `response.read(CHUNK)` into
a named temp file while counting bytes. If the next chunk would cross
`max_bytes`, it raises the existing 413 envelope before writing that chunk. After
a successful non-empty copy, it parses the temp file through
`_parse_deflection_submit_csv_file(...)`, preserving delimiter/BOM/warning
behavior from the shared CSV loader.

The existing `_read_bounded_https_blob(...)` stays for full-thread JSON and
other byte-oriented callers. Shared fetch status/error/cleanup behavior is kept
small and local so the security surface does not fork.

## Intentional

- This PR does not stream full-thread JSON blobs. The full-thread importer is
  already intentionally byte-oriented through
  `load_zendesk_full_thread_rows_from_json_bytes(...)`; changing that safely is a
  separate parser-design slice.
- This PR does not rewrite `load_source_rows_with_warnings_from_file()` into a
  streaming row iterator. It removes the extra blob `bytes` materialization while
  keeping the existing parser contract.
- The direct multipart upload helper remains unchanged; #1556 already moved
  that path to chunked temp-file staging.

## Deferred

- Follow-up hardening for #1458: stream rows out of the CSV parser instead of
  accumulating every parsed row before package construction, if live-volume
  profiling still shows memory pressure after the upload/blob-boundary fixes.
- Full-thread JSON follow-up: design a bounded streaming JSON artifact parser if
  full-thread exports approach the submit byte limit in live trials.

Parked hardening: none.

## Verification

- python -m py_compile extracted_content_pipeline/api/control_surfaces.py
  tests/test_extracted_content_deflection_submit.py
- python -m pytest tests/test_extracted_content_deflection_submit.py -q
  (58 passed)
- bash scripts/check_ascii_python.sh
- bash scripts/validate_extracted_content_pipeline.sh
- python extracted/_shared/scripts/forbid_atlas_reasoning_imports.py
  extracted_content_pipeline
- python scripts/audit_extracted_standalone.py --fail-on-debt
- bash scripts/run_extracted_pipeline_checks.sh (reasoning core: 295 passed;
  content pipeline: 4187 passed, 10 skipped)

## Estimated diff size

| File | LOC |
|---|---:|
| `extracted_content_pipeline/api/control_surfaces.py` | 156 |
| `plans/PR-Deflection-Blob-Upload-Tempfile.md` | 111 |
| `tests/test_extracted_content_deflection_submit.py` | 86 |
| **Total** | **353** |

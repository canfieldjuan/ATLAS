# PR-Deflection-Full-Thread-JSON-Tempfile

## Why this slice exists

#1556/#1561/#1562 moved CSV submit uploads and Blob CSV fetches off the old
single-buffer path and into bounded tempfile staging. The Zendesk full-thread
JSON path still takes the old shape: uploaded JSON reads `max_bytes + 1` at
once, Blob JSON reads `max_bytes + 1` at once, and the route hands a bytes
object to the thread normalizer. That leaves the submit boundary asymmetric
right where the live Zendesk export path now feeds private Blob artifacts.

This slice fixes the upstream submit boundary, not a downstream symptom: full
thread JSON is staged through the same bounded tempfile pattern as CSV before
parsing. The parser still loads JSON once it is time to parse; this PR does not
pretend to be a streaming JSON parser. It closes the upload/blob read shape so
large-but-allowed full-thread artifacts do not require a route-layer whole-file
read before normalization.

## Scope (this PR)

Ownership lane: content-ops/deflection-launch-readiness
Slice phase: Vertical slice

1. Add a file/path entry point for Zendesk full-thread JSON normalization.
2. Stage full-thread multipart JSON uploads into a bounded tempfile in chunks.
3. Stage full-thread Blob JSON fetches into a bounded tempfile with the existing
   pinned HTTPS fetch/SSRF guard, then parse from the file.
4. Add focused tests proving chunked upload reads, chunked Blob reads, unchanged
   metadata/output behavior, malformed JSON failure, empty failure, and oversize
   failure.

### Review Contract
- Acceptance criteria:
  - [ ] `importer_mode="full_thread"` multipart JSON no longer reads
        `max_bytes + 1` in one call before parsing.
  - [ ] `importer_mode="full_thread"` Blob JSON no longer reads
        `max_bytes + 1` in one call before parsing.
  - [ ] Existing public metadata remains stable: row counts, `uploaded_bytes` /
        `blob_bytes`, support platform, resolution evidence, status, CSAT,
        warnings, and private-note suppression.
  - [ ] Empty, oversize, malformed, and wrong-file/mode cases still fail closed
        with the existing public error shapes.
  - [ ] Tempfiles are cleaned up on success and failure.
- Affected surfaces: extracted Content Ops API submit loaders, Zendesk
  full-thread normalizer, submit tests, support-ticket normalizer tests.
- Risk areas: hidden whole-file reads, error-shape drift, tempfile leaks,
  private-note leakage, and accidental changes to CSV submit behavior.
- Reviewer rules triggered: R1, R2, R5, R8, R9, R10, R13, R14.

### Files touched

- `extracted_content_pipeline/api/control_surfaces.py`
- `extracted_content_pipeline/support_ticket_zendesk_thread.py`
- `plans/PR-Deflection-Full-Thread-JSON-Tempfile.md`
- `tests/test_extracted_content_deflection_submit.py`
- `tests/test_extracted_support_ticket_input_package.py`

## Mechanism

`support_ticket_zendesk_thread.py` gets
`load_zendesk_full_thread_rows_from_json_file(path)`, which reads and parses a
UTF-8 JSON artifact from an already-staged file, then delegates to the existing
`rows_from_zendesk_full_thread` normalizer.

The submit API keeps the existing mode contract. CSV remains the default. For
`importer_mode="full_thread"`, multipart uploads are copied into a temporary
JSON file using the same fixed-size upload chunk loop as CSV, then parsed via
the file entry point. Blob full-thread artifacts reuse the existing pinned HTTPS
fetch validation and response chunk copy into a temporary JSON file, then
parse via the same file entry point. Both paths return the existing
`(rows, byte_count, warnings)` tuple and clean up the tempfile in `finally`.

## Intentional

- This is not a streaming JSON parser. The correct safe upstream point for this
  slice is the submit boundary that was still reading full-thread artifacts into
  bytes before parsing. A true incremental parser would be a separate robustness
  slice with a dependency and broader parser contract.
- CSV behavior stays unchanged except for sharing generic copy helpers with
  JSON-specific error details.
- Blob oversize keeps the existing `deflection_submit_blob_too_large` reason so
  callers do not see an error-contract rename for private Blob submits.

## Deferred

- True streaming JSON parsing if live Zendesk artifacts approach the submit byte
  limit and route-layer tempfile staging is not enough.

Parked hardening: none.

## Verification

- Focused full-thread JSON submit tests -- 7 passed.
- Full deflection submit test file -- 62 passed.
- Extracted content pipeline validation script -- passed.
- Atlas reasoning import guard -- passed.
- Extracted standalone audit -- passed.
- ASCII Python check -- passed.
- Extracted sync script -- passed.
- Extracted pipeline CI check script -- 4192 passed, 10 skipped.

## Estimated diff size

| File | LOC |
|---|---:|
| `extracted_content_pipeline/api/control_surfaces.py` | 137 |
| `extracted_content_pipeline/support_ticket_zendesk_thread.py` | 14 |
| `plans/PR-Deflection-Full-Thread-JSON-Tempfile.md` | 111 |
| `tests/test_extracted_content_deflection_submit.py` | 99 |
| `tests/test_extracted_support_ticket_input_package.py` | 14 |
| **Total** | **375** |

# PR-Deflection-CSV-Parser-Memory-Hardening

## Why this slice exists

Issue #1458 is still open after current-main recheck. The HTTP upload and blob
staging edges already stream into temp files, but the shared CSV parser used by
the deflection report still loads the whole file as bytes, decodes the whole
file to a text string, feeds that text through `StringIO`, builds a full raw-row
list, and only then builds the returned row list. That keeps avoidable duplicate
copies resident on large CSV uploads before the report can apply its downstream
row cap.

The operator chose a three-step sequence: first parser memory hardening, then
parser error UX, then parser API widening. This PR is the first step only.

This is over the 400 LOC soft cap because the parser already supports BOM,
UTF-16 inference, UTF-8 recovery, CP1252/Latin-1 fallback, replacement-character
warnings, and mixed-delimiter rejection. Preserving those behaviors while
removing whole-file byte/text/raw-row materialization requires a small bounded
encoding scan and streaming consistency state rather than a one-line file-handle
swap.

## Scope (this PR)

Ownership lane: content-ops/deflection-report
Slice phase: Production hardening

1. Stream the shared CSV parser internals from the source file instead of
   materializing full-file bytes, full-file text, and a separate raw-row list.
2. Preserve the existing parser API shape and caller behavior: callers still get
   `(rows, warnings)`, and the deflection submit endpoint keeps its current
   request/response contract.
3. Keep delimiter, header, encoding, leading-row warning, and inconsistent-row
   behavior covered by the existing parser tests.
4. Add focused regression coverage proving the CSV source adapter path no
   longer depends on whole-file byte reads and still parses representative
   support-ticket CSV rows.

### Review Contract
- Acceptance criteria:
  - [ ] The CSV parser no longer calls full-file byte loading for CSV inputs.
  - [ ] The parser no longer builds a separate full raw-row list before building
        returned rows.
  - [ ] Existing CSV behavior remains compatible for delimiter detection,
        skipped prologue warnings, UTF-8/UTF-16/legacy encoding handling, and
        inconsistent column rejection.
  - [ ] The deflection submit upload/blob staging code remains unchanged except
        for tests that prove the shared parser path.
- Affected surfaces: API input parsing, extracted package parser internals,
  deflection report uploads.
- Risk areas: performance, backcompat, user-input parsing, encoding behavior.
- Reviewer rules triggered: R1, R2, R3, R5, R6, R7, R10, R12, R13, R14.

### Files touched

- `extracted_content_pipeline/campaign_customer_data.py`
- `plans/PR-Deflection-CSV-Parser-Memory-Hardening.md`
- `tests/test_extracted_campaign_source_adapters.py`

## Mechanism

Replace the whole-file CSV read path with bounded-memory passes over the same
temp file:

1. Inspect bytes in chunks to choose a decoding plan. The inspection keeps only
   counters and a small prefix sample needed for BOM, UTF-16 NUL-pattern, UTF-8
   validity, legacy fallback, replacement-character warning, and ambiguity
   warning decisions.
2. Decode only a bounded prefix sample for the fast delimiter/header path when
   the sample contains a real header hint. If the prefix does not prove the
   header, stream the file across candidate dialects so large provider
   preambles can still be skipped before the real header.
3. Reopen the file as a text stream with the selected decoding plan and feed the
   file handle directly to `csv.reader`.
4. Find the header while iterating, store only leading rows needed for the
   existing skipped-prologue warning, then build returned rows while validating
   column consistency in the same pass.

The parser still returns a list because downstream callers and report metadata
currently depend on the existing contract. This PR removes duplicate full-file
copies; it does not yet make report limits parser-level caps.

## Intentional

- No parser API widening in this PR. Returning a richer result with total counts,
  included counts, and truncation metadata is useful, but it changes caller
  contracts and belongs after memory hardening and error UX.
- No user-facing parse-error redesign in this PR. Current CSV parse failures
  still surface through the existing 400 path; structured safe error details are
  the next PR.
- JSON and JSONL source loading are left unchanged. Issue #1458 is about the CSV
  upload parser used by deflection reports.

## Deferred

- Parser error UX: return safe structured CSV parse failures with actionable
  user guidance instead of only the generic upload/blob parse message.
- Parser API widening: open a separate GitHub issue when ready to add a richer
  parser result contract for parser-level caps, counts, and truncation metadata.

Parked hardening: none.

## Verification

- Python compile check for the parser module -- passed.
- Source-adapter parser test file -- 119 passed.
- Combined source-adapter parser and deflection submit test files -- 181 passed.
- Extracted pipeline check script -- reasoning core 295 passed;
  extracted pipeline 4,400 passed, 10 skipped.

## Estimated diff size

| File | LOC |
|---|---:|
| `extracted_content_pipeline/campaign_customer_data.py` | 682 |
| `plans/PR-Deflection-CSV-Parser-Memory-Hardening.md` | 118 |
| `tests/test_extracted_campaign_source_adapters.py` | 90 |
| **Total** | **890** |

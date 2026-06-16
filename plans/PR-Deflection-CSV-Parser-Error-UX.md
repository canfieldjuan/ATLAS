# PR-Deflection-CSV-Parser-Error-UX

## Why this slice exists

Issue #1458 now has the agreed three-step parser sequence. PR #1610 closed step
1 by removing avoidable full-file CSV parser materialization while preserving the
current parser API. The next user-visible gap is error clarity: the shared CSV
loader can tell when a file has no header, cannot be decoded safely, or has
inconsistent columns, but the deflection submit surface collapses those failures
to generic "could not be parsed" strings. A buyer or operator can tell the
upload failed, but not what to fix in the file.

This slice keeps the fix upstream of the symptom. It classifies the parser's
own failure modes at the shared CSV loader boundary, then translates those safe
codes into deflection submit HTTP details. That avoids adding string matching in
the API layer and keeps parser API widening deferred.

The final diff is slightly over the 400-line soft cap because review found two
same-scope detection gaps: malformed BOM-prefixed encoding failures and the
delimiter-consistency branch both needed direct regressions. Splitting those
tests out would leave the parser error UX contract partially unproved.

## Scope (this PR)

Ownership lane: content-ops/deflection-report
Slice phase: Production hardening

1. Add safe structured CSV parse errors for the deflection-relevant parser
   classes: missing header, encoding failure, and inconsistent column counts.
2. Return those parse errors from deflection CSV upload and blob submit paths as
   structured 400 details with a stable code, safe user-facing message, and
   "how to fix" guidance.
3. Preserve the existing shared loader API for successful parses and keep
   parser API widening out of scope.
4. Add focused parser and submit-surface regressions proving each new error
   branch fires without exposing raw uploaded row contents.

### Files touched

- `extracted_content_pipeline/api/control_surfaces.py`
- `extracted_content_pipeline/campaign_customer_data.py`
- `plans/PR-Deflection-CSV-Parser-Error-UX.md`
- `tests/test_extracted_campaign_source_adapters.py`
- `tests/test_extracted_content_deflection_submit.py`

### Review Contract

- Acceptance criteria:
  - [ ] Missing-header CSV input reaches deflection submit as a structured
        `csv_missing_header` 400 response with correction guidance.
  - [ ] Unsafe or undecodable CSV input reaches deflection submit as a
        structured `csv_encoding_error` 400 response with correction guidance.
  - [ ] Inconsistent-column CSV input reaches deflection submit as a structured
        `csv_inconsistent_columns` 400 response with correction guidance.
  - [ ] Structured parse details do not include raw row values, uploaded text,
        email addresses, phone numbers, or other user-provided field contents.
  - [ ] Successful CSV parsing, load warnings, row output shape, and existing
        caller exception compatibility remain unchanged.
- Affected surfaces: API, file upload, blob import, extracted package parser,
  tests, CI enrollment.
- Risk areas: user input safety, backcompat, error handling, parser correctness.
- Reviewer rules triggered: R1, R2, R3, R5, R6, R10, R12, R14.

## Mechanism

Introduce a small structured parse error type in the shared CSV loader with a
stable code, safe message, safe fix guidance, and optional non-sensitive row
index metadata. It should inherit from `ValueError` so existing non-HTTP callers
that catch parse failures keep working.

Raise that structured error at the parser's existing decision points instead of
raising plain text only:

- no plausible header row, or an empty header after delimiter selection;
- encoding scans that fail closed on undecodable or NUL-heavy text;
- delimiter and row-width validation that finds inconsistent column counts.

The deflection submit CSV parser wrapper then catches the structured type before
the generic parser exception fallback and returns an HTTP 400 detail object. The
detail object carries a deflection-specific reason plus the parser code,
message, fix guidance, and optional row index. The generic fallback remains for
unexpected parser exceptions.

## Intentional

- No parser API widening in this slice. Successful callers still receive the
  existing row and warning tuple, and parser-level cap/count metadata remains a
  later design issue.
- No raw CSV snippets in error details. Even if a preview would help debugging,
  the submit path is public input and the safe response should only include
  codes, guidance, and numeric row metadata.
- JSON and Zendesk full-thread parse errors stay out of scope. The user request
  and #1458 are about CSV upload parser failures.
- The shared parser still raises a `ValueError` subclass instead of a separate
  result envelope so existing callers do not need a compatibility migration.

## Deferred

- Parser API widening remains deferred until after parser memory hardening and
  parser error UX. When reached, open the separate GitHub issue the operator
  requested for richer parser caps/counts/truncation metadata.
- Ingestion-file and CLI presentation of the same structured parse errors is
  deferred unless required by tests; this slice hardens the deflection submit
  surface first.

Parked hardening: none.

## Verification

- Focused pytest for the campaign source adapter and deflection submit suites
  passed with 190 tests after the review fixes.
- The extracted pipeline check bundle passed: extracted reasoning core checks
  passed with 295 tests, then extracted content pipeline checks passed with
  4,409 tests and 10 skipped after the review fixes.
- Local PR review passed with the PR body file supplied.

## Estimated diff size

| File | LOC |
|---|---:|
| `extracted_content_pipeline/api/control_surfaces.py` | 14 |
| `extracted_content_pipeline/campaign_customer_data.py` | 112 |
| `plans/PR-Deflection-CSV-Parser-Error-UX.md` | 126 |
| `tests/test_extracted_campaign_source_adapters.py` | 91 |
| `tests/test_extracted_content_deflection_submit.py` | 87 |
| **Total** | **430** |

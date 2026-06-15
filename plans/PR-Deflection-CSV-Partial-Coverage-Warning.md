# PR-Deflection-CSV-Partial-Coverage-Warning

## Why this slice exists

#1467 still has one admission gap after #1573 and #1575: accepted source-row
CSV uploads can hide partial row loss. #1575 rejects the total failure case
where a non-empty source-row CSV yields zero usable rows, but a file with at
least one usable row still serializes `ACCEPT` even when other rows were
dropped. The diagnostics expose raw/usable counts, but there is no explicit
warning saying the accepted upload was partial.

This slice adds a warning-only boundary for accepted-but-partial CSV source
uploads. It does not invent a low-coverage reject threshold; it makes row loss
visible while leaving threshold policy to a later product decision.

## Scope (this PR)

Ownership lane: content-ops/deflection-parser-admission
Slice phase: Production hardening

1. Add a `coverage_warnings` list under `source_row_admission` when a
   source-row CSV has raw rows greater than usable rows and at least one usable
   row.
2. Report a stable warning code, location, raw count, usable count, skipped
   count, and usable ratio without changing `admission_decision`.
3. Add focused coverage for accepted-but-partial CSVs, clean accepted CSVs,
   zero-usable rejects, and non-CSV compatibility.

### Files touched

- `extracted_content_pipeline/campaign_source_adapters.py`
- `plans/PR-Deflection-CSV-Partial-Coverage-Warning.md`
- `tests/test_extracted_content_ingestion_diagnostics.py`

### Review Contract

- Acceptance criteria:
  - [ ] Accepted source-row CSV uploads with skipped rows serialize a
        `coverage_warnings` entry with code `partial_source_row_coverage`.
  - [ ] The warning includes location `source_row_csv`, raw source row count,
        usable source row count, skipped source row count, and usable source
        ratio.
  - [ ] Clean accepted source-row CSV uploads do not serialize coverage
        warnings.
  - [ ] Zero-usable source-row CSV uploads keep the existing `REJECT` decision
        and do not need a duplicate partial-coverage warning.
  - [ ] JSON/JSONL and opportunity-row inspection payloads remain backwards
        compatible and do not emit source-row CSV warning policy.
- Affected surfaces: extracted package diagnostics, source-row CSV inspection,
  CLI/API JSON response payloads.
- Risk areas: backwards compatibility, false diagnostic precision, policy
  threshold drift, CI enrollment.
- Reviewer rules triggered: R1, R2, R5, R10, R12, R13, R14.

## Mechanism

`SourceRowAdmissionDiagnostics` already owns the input format, raw row count,
usable row count, and ratio. This PR adds a derived warning list beside the
derived admission decision:

- source-row CSV and raw rows > usable rows > 0 -> one warning
- source-row CSV and usable rows == raw rows -> no warning
- source-row CSV and usable rows == 0 -> existing reject decision, no duplicate
  partial warning
- non-CSV inputs -> no source-row CSV warning object

The warning is nested under `source_row_admission`, so clients that ignore
unknown keys remain compatible. It is warning-only and does not change `ok`,
`admission_decision`, import behavior, or route status codes.

## Intentional

- No low-coverage reject threshold in this PR. Any partial accepted upload gets
  a warning; whether a 10%, 25%, or 50% usable ratio should reject is still a
  product decision.
- No route-level behavior change. This is diagnostics enrichment only.
- No streaming or upload-buffering changes. #1458 owns that lane.

## Deferred

- #1467 low-coverage reject threshold: decide whether warning-only partial
  coverage is enough, or whether a product-backed threshold should hard reject.
- #1458 streaming upload memory hardening remains separate.

Parked hardening: none.

## Verification

- Focused ingestion diagnostics tests: 14 passed.
- Adjacent source-adapter plus diagnostics tests: 130 passed.
- Extracted package validation: passed.
- Standalone audit: 0 findings.
- Reasoning-import guard: clean.
- ASCII Python check: passed.
- Extracted pipeline checks: 4280 passed, 10 skipped.
- Pending before push: local PR review/pre-push hook.

## Estimated diff size

| File | LOC |
|---|---:|
| `extracted_content_pipeline/campaign_source_adapters.py` | 23 |
| `plans/PR-Deflection-CSV-Partial-Coverage-Warning.md` | 105 |
| `tests/test_extracted_content_ingestion_diagnostics.py` | 39 |
| **Total** | **167** |

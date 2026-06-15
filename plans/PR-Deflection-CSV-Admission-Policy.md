# PR-Deflection-CSV-Admission-Policy

## Why this slice exists

#1467 asks for an explicit parser-admission boundary so a non-empty upload
cannot silently become zero useful rows. #1573 added the deterministic evidence
surface: source-row CSV inspection now reports raw rows, usable rows, usable
ratio, mapped fields, ignored private/internal fields, and populated unmapped
fields. The remaining product gap is policy: the report still exposes those
facts as diagnostics only, so callers have to infer whether the upload is
accepted or rejected.

This slice adds the first hard boundary at the safest point: a non-empty
source-row CSV with zero usable source rows is rejected with a stable reason
and location. It does not guess at low-coverage thresholds yet.

## Scope (this PR)

Ownership lane: content-ops/deflection-parser-admission
Slice phase: Production hardening

1. Add an explicit source-row CSV admission decision to the existing
   diagnostics payload: `ACCEPT` for usable uploads, `REJECT` for non-empty
   CSV source uploads with zero usable rows, and a stable reason/location for
   the reject case.
2. Keep import behavior compatible with the existing `ok=false` gate, while
   making the reject reason machine-readable in file inspect and file import
   diagnostics.
3. Add focused coverage for zero-usable rejection, usable CSV acceptance, empty
   CSV/no-data behavior, and non-CSV compatibility.

### Files touched

- `extracted_content_pipeline/campaign_source_adapters.py`
- `plans/PR-Deflection-CSV-Admission-Policy.md`
- `tests/test_extracted_content_ingestion_diagnostics.py`

### Review Contract

- Acceptance criteria:
  - [ ] Non-empty source-row CSV uploads with zero usable rows serialize a
        `REJECT` decision with reason `no_usable_source_rows` and location
        `source_row_csv`.
  - [ ] The reject payload includes raw row count, usable row count, usable
        ratio, and populated unmapped fields from the existing diagnostics.
  - [ ] Source-row CSV uploads with at least one usable row serialize `ACCEPT`
        and keep existing warnings as warnings.
  - [ ] Opportunity-row, JSON, and JSONL inspection payloads remain backwards
        compatible and do not emit source-row CSV policy decisions.
  - [ ] Existing import rejection behavior remains `ingestion_not_ready`, with
        the new decision nested in diagnostics rather than a new route
        contract.
- Affected surfaces: extracted package diagnostics, source-row CSV inspection,
  file import diagnostics, CLI/API JSON response payloads.
- Risk areas: backwards compatibility, false rejection, diagnostic contract
  drift, CI enrollment.
- Reviewer rules triggered: R1, R2, R5, R10, R12, R13, R14.

## Mechanism

The policy lives beside the source-row CSV diagnostics created in #1573. The
diagnostic object derives a small admission decision from facts it already
owns: input format, raw source row count, and usable source row count.

For this slice the hard rule is deliberately narrow:

- source-row CSV and raw rows > 0 and usable rows == 0 -> `REJECT`
- source-row CSV and usable rows > 0 -> `ACCEPT`
- raw rows == 0 -> no hard source-row CSV reject reason; the existing empty
  file/header behavior continues to govern
- non-CSV inputs -> no source-row CSV policy object

`IngestionDiagnosticsReport.as_dict` continues to emit the optional
`source_row_admission` section only when CSV source-row diagnostics exist. The
new decision is nested inside that section so clients that ignore unknown keys
continue to work, while newer callers can show a clear reject reason and
location.

## Intentional

- No low-coverage threshold in this PR. A 1-of-100 usable upload may deserve a
  future reject or warning policy, but setting that threshold is a product
  decision and needs real upload evidence.
- No streaming or file-buffering changes. #1458 owns upload memory hardening.
- No route-level error code change. File import already rejects `ok=false` as
  `ingestion_not_ready`; this PR enriches diagnostics instead of breaking that
  contract.

## Deferred

- #1467 low-coverage policy: decide whether non-zero but low usable source
  ratios should warn only or reject, and set a threshold with fixture evidence.
- #1458 streaming upload memory hardening remains separate.

Parked hardening: none.

## Verification

- Focused ingestion diagnostics tests: 13 passed.
- Adjacent source-adapter plus diagnostics tests: 129 passed.
- Extracted package validation: passed.
- Standalone audit: 0 findings.
- Reasoning-import guard: clean.
- ASCII Python check: passed.
- Extracted pipeline checks: 4279 passed, 10 skipped.
- Pending before push: local PR review/pre-push hook.

## Estimated diff size

| File | LOC |
|---|---:|
| `extracted_content_pipeline/campaign_source_adapters.py` | 18 |
| `plans/PR-Deflection-CSV-Admission-Policy.md` | 115 |
| `tests/test_extracted_content_ingestion_diagnostics.py` | 31 |
| **Total** | **164** |

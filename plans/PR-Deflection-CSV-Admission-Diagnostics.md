# PR-Deflection-CSV-Admission-Diagnostics

## Why this slice exists

#1467 asks for a parser-admission boundary that does not silently turn a
non-empty upload into zero useful rows. #1457 closed the first Zendesk header
admission gap in #1571, but the remaining operator-facing gap is still
diagnostic: when a CSV has populated columns that the source-row adapter does
not understand, the current inspect output only reports row warnings such as
`missing_source_text`. It does not say which headers mapped to source id,
title, customer-visible text, private/internal text, or which populated columns
were ignored.

This slice adds the deterministic diagnostic surface first. It keeps ingestion
permissive, but gives the existing inspect path enough structured evidence to
make the later ACCEPT/REJECT boundary a small policy change instead of another
heuristic rewrite.

The final diff is slightly over the 400 LOC target because the Codex review
surfaced a class bug in wide CSV alias classification; the regression keeps
more than 25 mapped aliases from being mislabeled as unmapped.

## Scope (this PR)

Ownership lane: content-ops/deflection-parser-admission
Slice phase: Production hardening

1. Add source-row CSV admission diagnostics that report mapped source fields,
   ignored private/internal fields, populated unmapped fields, raw source row
   count, usable source row count, and usable source ratio.
2. Surface those diagnostics through the existing ingestion inspection report
   and CLI/API JSON payloads without changing import acceptance behavior.
3. Add regression coverage for a CSV with an unmapped populated body column,
   a CSV with mapped Zendesk public/private columns, and a non-CSV source-row
   input that must not invent CSV-only diagnostics.

### Files touched

- `extracted_content_pipeline/campaign_source_adapters.py`
- `extracted_content_pipeline/ingestion_diagnostics.py`
- `plans/PR-Deflection-CSV-Admission-Diagnostics.md`
- `tests/test_extracted_content_ingestion_diagnostics.py`

### Review Contract

- Acceptance criteria:
  - [ ] Source-row CSV inspection reports raw row count, usable source count,
        usable ratio, mapped source fields, ignored private fields, and
        populated unmapped fields.
  - [ ] A non-empty CSV whose customer text lives only in an unknown populated
        body column keeps the existing row-level warning and now names that
        column in diagnostics.
  - [ ] Zendesk public comment/history/private-note columns are classified in
        diagnostics consistently with #1571 extraction behavior.
  - [ ] JSON/JSONL and opportunity-row inspection payloads remain backwards
        compatible and do not emit misleading CSV header diagnostics.
- Affected surfaces: extracted package diagnostics, source-row CSV inspection,
  CLI/API JSON response payloads.
- Risk areas: backwards compatibility, false diagnostic precision, private text
  handling, CI enrollment.
- Reviewer rules triggered: R1, R2, R5, R10, R12, R13, R14.

## Mechanism

The source adapter already owns the alias sets that decide whether a source
row has customer-visible text, source id, title, or private/internal text. This
PR reuses those sets to classify the keys present in CSV-loaded source rows.
The classifier produces an immutable diagnostics object with counts and bounded
field-name lists; it does not read files itself and does not alter the
normalized opportunities.

`inspect_ingestion_file` attaches that diagnostics object only for
`source_rows=true` CSV inspection. `inspect_ingestion_rows` stays generic
because inline rows do not have a CSV header admission problem. The report
serializes the diagnostic object as optional `source_row_admission`, so
existing clients keep the current fields and new clients can show mapped versus
unmapped column evidence.

## Intentional

- This PR does not reject uploads yet. It makes the evidence explicit first,
  preserving current import compatibility while setting up #1467's final policy
  boundary.
- The classifier is source-adapter-owned rather than CSV-reader-owned because
  the low-level CSV reader is shared by opportunity imports and has no product
  knowledge about source text, Zendesk privacy, or support-ticket semantics.
- Diagnostics report field names and counts only; it does not copy field
  values, so private/internal note content is not exposed in the report.

## Deferred

- #1467 final admission policy: convert these diagnostics into an explicit
  ACCEPT/REJECT boundary for non-empty uploads with zero or too-low usable
  source coverage.
- #1458 streaming upload memory hardening remains separate; this PR does not
  change upload buffering.

Parked hardening: none.

## Verification

- Focused ingestion diagnostics tests: 12 passed.
- Adjacent source-adapter plus diagnostics tests: 128 passed.
- Extracted package validation: passed.
- Standalone audit: 0 findings.
- Reasoning-import guard: clean.
- ASCII Python check: passed.
- Extracted pipeline checks: 4278 passed, 10 skipped.
- Pending before push: local PR review/pre-push hook.

## Estimated diff size

| File | LOC |
|---|---:|
| `extracted_content_pipeline/campaign_source_adapters.py` | 118 |
| `extracted_content_pipeline/ingestion_diagnostics.py` | 57 |
| `plans/PR-Deflection-CSV-Admission-Diagnostics.md` | 119 |
| `tests/test_extracted_content_ingestion_diagnostics.py` | 157 |
| **Total** | **451** |

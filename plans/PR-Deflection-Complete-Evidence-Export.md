# PR-Deflection-Complete-Evidence-Export

## Why this slice exists

#1588 says the paid deflection report needs separate surfaces: concise hosted
view, curated PDF, and a complete evidence export. #1590 merged the hosted
view, so this slice builds the prerequisite that keeps future web/PDF
caps honest: an uncapped structured evidence export derived from the same
deterministic FAQ result data as the paid report.

Without the export, later PDF/report-polish slices would have to either keep
every evidence row inline or weaken the completeness promise. This PR gives
later renderers a stable export contract first, while avoiding the #1590
portfolio files.

This is over the 400 LOC soft cap because the artifact projection, CLI writer,
contract docs, and regression tests need to land together. Splitting the CLI or
tests from the projection would create an export contract without a reproducible
operator artifact path or failure proof.

## Scope (this PR)

Ownership lane: content-ops/deflection-launch-readiness
Slice phase: Vertical slice

1. Add a `deflection_evidence.v1` JSON export to the paid deflection artifact.
2. Project existing FAQ item data into stable question rows and evidence rows:
   question id/rank, source ticket ids, evidence quote, answer classification,
   answer linkage, term mappings, and outcome diagnostics.
3. Add a CLI flag that writes only the evidence export JSON for validation and
   future attachment/download wiring.
4. Update the report promise text so completeness is satisfied by the export,
   not by forcing every evidence row into inline web/PDF content.
5. Do not touch paid result-page files or add a new public download route in
   this slice.

### Files touched

- `docs/extraction/validation/fixtures/deflection_resolution_evidence_live_proof_20260609/report.md`
- `docs/frontend/content_ops_faq_deflection_report_example.json`
- `docs/frontend/content_ops_faq_report_contract.md`
- `extracted_content_pipeline/faq_deflection_report.py`
- `plans/PR-Deflection-Complete-Evidence-Export.md`
- `scripts/build_content_ops_deflection_report.py`
- `tests/test_content_ops_deflection_report.py`

### Review Contract

- Acceptance criteria:
  - [ ] Paid artifact dictionaries include a structured `evidence_export` with
        schema version `deflection_evidence.v1`.
  - [ ] The export is derived from existing structured FAQ result items, not
        parsed from Markdown.
  - [ ] Source ids and evidence quotes are preserved in evidence rows, including
        rows that have source ids but no rendered quote.
  - [ ] Publishable and no-proven-answer questions keep distinct answer linkage
        and classification fields.
  - [ ] The CLI can write the evidence export JSON independently of Markdown.
  - [ ] Promise text no longer implies inline web/PDF content is the uncapped
        completeness surface.
- Affected surfaces: extracted deflection artifact shape, local report CLI,
  report contract docs, focused report tests.
- Risk areas: persisted artifact compatibility, evidence row truncation, future
  renderer contract drift, and collision with #1590 portfolio rendering work.
- Reviewer rules triggered: R1, R2, R5, R9, R10, R12, R14.

## Mechanism

The deflection artifact already stores `faq_result.items` as structured data.
This slice adds a pure projection over those items. Each ranked question gets a
stable export question id, answer status, answer linkage, source ids,
publishable answer/steps when present, term mappings, and outcome diagnostics.
Evidence rows are then emitted from the same items with stable row ids and
source-id/quote fields.

The projection is added to the artifact dictionary so paid artifact consumers
can retrieve it after unlock. The CLI also gains an evidence-output path that
writes only the export JSON, giving operator proofs and future download wiring a
single contract to consume.

## Intentional

- No new portfolio route in this PR. #1590 owns the paid result-page rendering
  surface, and a later download route can expose the already-persisted export.
- No PDF redesign here. #1588 explicitly says the curated PDF should wait until
  the complete export exists.
- No Markdown parsing. Markdown remains a renderer; the export is built from
  the structured FAQ item data.
- Evidence rows may have an empty quote when a source id exists but the current
  rendered FAQ item did not include a quote for that source. The row still
  preserves the source id so the export is not silently truncated to rendered
  quote count.

## Deferred

- #1590 merged the hosted paid result-page consolidated view.
- #1588 follow-up: public download route/button for the evidence export after
  the paid web surface settles.
- #1588 follow-up: curated/shareable PDF that links to or attaches this export
  instead of trying to inline every evidence row.
- #1588 follow-up: full structured `deflection.v1` paid report model.

Parked hardening: none.

## Verification

- `python -m pytest tests/test_content_ops_deflection_report.py -q`
  - Result: included in focused 54-test rerun after review fixes.
- `python -m pytest tests/test_content_ops_faq_report_contract_docs.py tests/test_content_ops_deflection_resolution_live_proof.py -q`
  - Result: included in focused 54-test rerun after review fixes.
- `python -m pytest tests/test_content_ops_deflection_report.py tests/test_content_ops_faq_report_contract_docs.py tests/test_content_ops_deflection_resolution_live_proof.py -q`
  - Result: 54 passed.
- `scripts/run_extracted_pipeline_checks.sh` via `bash`
  - Result: 4306 passed, 10 skipped.
- Python compile check for changed Python files.
  - Result: passed.
- Local PR review bundle.
  - Pending via push wrapper before push.

## Estimated diff size

| File | LOC |
|---|---:|
| `docs/extraction/validation/fixtures/deflection_resolution_evidence_live_proof_20260609/report.md` | 2 |
| `docs/frontend/content_ops_faq_deflection_report_example.json` | 172 |
| `docs/frontend/content_ops_faq_report_contract.md` | 58 |
| `extracted_content_pipeline/faq_deflection_report.py` | 155 |
| `plans/PR-Deflection-Complete-Evidence-Export.md` | 131 |
| `scripts/build_content_ops_deflection_report.py` | 13 |
| `tests/test_content_ops_deflection_report.py` | 177 |
| **Total** | **708** |

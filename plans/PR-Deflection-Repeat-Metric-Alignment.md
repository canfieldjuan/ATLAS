# PR-Deflection-Repeat-Metric-Alignment

## Why this slice exists

Issue #1481 is the buyer-visible symptom of the old category-sized repeat
counting path: the snapshot and paid report called broad bucket membership
"repeat-ticket hits" and priced it as repeat work. The upstream clustering
foundation is now corrected by #1531, and the report helper already excludes
single-ticket questions from `repeat_ticket_count`.

This vertical slice aligns the buyer-facing copy with that corrected meaning:
the count is question-level repeat tickets, not arbitrary category membership
or every uploaded row. The persisted snapshot field stays named
`repeat_ticket_count` for compatibility, but labels and explanatory copy should
tell the buyer exactly what the number means.

## Scope (this PR)

Ownership lane: deflection/clustering
Slice phase: Vertical slice

1. Update paid report Support Tax copy to say question-level repeat tickets and
   clarify that only tickets in repeated question clusters are cost-sized.
2. Update portfolio result-page metric labels and repeat-volume diagnostic copy
   to use the same question-level wording.
3. Keep the JSON field names and API contracts unchanged.
4. Add/adjust tests so stale "repeat-ticket hits" copy cannot reappear on the
   paid report or hosted result page.

### Review Contract

- Acceptance criteria:
  - [ ] Paid report Support Tax wording uses question-level repeat-ticket
        language and no longer says "repeat-ticket hits."
  - [ ] Server-rendered and React fallback result pages use the same
        question-level repeat-ticket label and diagnostic copy.
  - [ ] Snapshot/portfolio JSON contracts remain backward compatible:
        `summary.repeat_ticket_count` is still required and parsed as before.
  - [ ] Existing zero/light/ready repeat-volume states still render.
- Affected surfaces: extracted deflection report Markdown, portfolio result
  page server renderer, portfolio React fallback, and tests.
- Risk areas: buyer-facing claim precision, frontend copy drift, and response
  contract compatibility.
- Reviewer rules triggered: R1, R2, R5, R9, R10, R12, R14.

### Files touched

- `HARDENING.md`
- `docs/extraction/validation/fixtures/deflection_resolution_evidence_live_proof_20260609/report.md`
- `docs/frontend/content_ops_faq_deflection_report_example.json`
- `extracted_content_pipeline/faq_deflection_report.py`
- `plans/PR-Deflection-Repeat-Metric-Alignment.md`
- `portfolio-ui/api/content-ops/deflection/result-page.js`
- `portfolio-ui/scripts/faq-deflection-result-page.test.mjs`
- `portfolio-ui/src/pages/FaqDeflectionResult.tsx`
- `tests/test_content_ops_deflection_report.py`
- `tests/test_content_ops_deflection_resolution_live_proof.py`

## Mechanism

The implementation keeps `repeat_ticket_count` as the data contract and changes
only the presentation layer around that field. The paid report still uses the
existing repeat-ticket count for cost sizing, but its copy names the count as
question-level repeat tickets and points out that single-ticket questions are
excluded. The portfolio result page reads the same field as before, but its
metric label and diagnostic copy mirror the report terminology.

## Intentional

- No JSON schema rename. Renaming `repeat_ticket_count` would break stored
  snapshots and hosted page parsing for no product benefit.
- No new clustering logic. #1531 fixed the question-level lexical foundation;
  this slice is the buyer-facing presentation layer.
- No live CFPB re-baseline in this PR. That needs the live artifact/input and
  should be recorded separately once run.

## Deferred

- Live CFPB artifact re-baseline after this wording lands.
- Embedding booster from #1504 for same-meaning questions with low lexical
  overlap.
- Issue #1518 status vocabulary bug.

Parked hardening:
- `portfolio-ui npm audit vulnerabilities` - existing dependency audit findings
  are security debt but require package upgrades outside this copy slice.
- `Customer-wording FAQ headings can publish raw PII` and
  `Safe-vocabulary representative label collisions render duplicate FAQ
  headings` remain parked because they are renderer hardening, not
  repeat-metric presentation alignment.

## Verification

- Command passed: python -m py_compile extracted_content_pipeline/faq_deflection_report.py tests/test_content_ops_deflection_report.py.
- Command passed: python -m py_compile tests/test_content_ops_deflection_resolution_live_proof.py.
- Command passed: pytest tests/test_content_ops_deflection_resolution_live_proof.py -q - 3 passed.
- Command passed: pytest tests/test_content_ops_deflection_report.py -q - 37 passed.
- Command passed: pytest tests/test_content_ops_faq_report_contract_docs.py::test_content_ops_faq_deflection_example_matches_producer_shape tests/test_content_ops_deflection_resolution_live_proof.py::test_resolution_live_proof_regenerates_from_committed_csv tests/test_content_ops_deflection_report.py -q - 39 passed.
- Command passed: cd portfolio-ui && npm run test:deflection-result.
- Command passed: cd portfolio-ui && npm run test:deflection-atlas-proxy.
- Command passed: cd portfolio-ui && npm run build.
- Command passed: bash scripts/validate_extracted_content_pipeline.sh.
- Command passed: python extracted/_shared/scripts/forbid_atlas_reasoning_imports.py extracted_content_pipeline.
- Command passed: python scripts/audit_extracted_standalone.py --fail-on-debt.
- Command passed: bash scripts/check_ascii_python.sh.
- Command passed: bash scripts/run_extracted_pipeline_checks.sh - 4066 passed, 10 skipped, 1 warning.
- Pending before push: local PR review.

## Estimated diff size

| File | LOC |
|---|---:|
| `HARDENING.md` | 9 |
| `docs/extraction/validation/fixtures/deflection_resolution_evidence_live_proof_20260609/report.md` | 4 |
| `docs/frontend/content_ops_faq_deflection_report_example.json` | 2 |
| `extracted_content_pipeline/faq_deflection_report.py` | 11 |
| `plans/PR-Deflection-Repeat-Metric-Alignment.md` | 123 |
| `portfolio-ui/api/content-ops/deflection/result-page.js` | 10 |
| `portfolio-ui/scripts/faq-deflection-result-page.test.mjs` | 11 |
| `portfolio-ui/src/pages/FaqDeflectionResult.tsx` | 10 |
| `tests/test_content_ops_deflection_report.py` | 9 |
| `tests/test_content_ops_deflection_resolution_live_proof.py` | 8 |
| **Total** | **197** |

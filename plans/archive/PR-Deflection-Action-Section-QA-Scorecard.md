# PR-Deflection-Action-Section-QA-Scorecard

## Why this slice exists

#1612's current report-actionability arc has landed the model contract
and the PDF renderer for the new action sections. PR #1758 explicitly deferred
S4C: add cross-surface assertions proving renderer output agrees with the paid
model while Snapshot stays closed. PR #1760 made the PDF render the action
sections, but the shared QA scorecard still only counts/caps the older ranked
question, question-detail, SEO, and diagnostic sections.

Root cause: `_scorecard_counts`, `_scorecard_section_total`, and the PDF/export
artifact observer do not know the action-section row totals, and the PDF
validator previously treated action headings as unconditional document markers
instead of model-driven sections. That can either miss a PDF that drops
`priority_fix_queue`, `top_unresolved_repeats`, `drafted_resolutions`, or
`already_covered_still_recurring`, or falsely fail older/live artifacts whose
model has action sections with no rows.

This slice fixes the root in the QA contract, not the renderer: the scorecard
requires the action sections in the paid model, learns their totals/caps for
PDF, and the PDF/export validator observes action-section rows from rendered
PDF text only when the model has rows for that section.

Diff budget note: this slice is over the 400 LOC soft cap because the review
fix has to keep the scorecard contract, PDF artifact checker, regression tests,
and plan/body reconciliation in one PR. Splitting the checker from the tests
would leave the privacy/report-delivery gate under-proven.

## Scope (this PR)

Ownership lane: issue-1612/deflection-full-report-delivery-actionability
Slice phase: Functional validation

1. Add action-section row totals to the shared full-report QA scorecard counts
   and require the action sections in the paid report model.
2. Teach the PDF/export artifact validator to observe visible action-section
   rows in the rendered PDF text without requiring empty action sections.
3. Add focused regression tests proving missing action-section rows fail the
   scorecard/validator, action rows do not inflate ranked-row observations, and
   result-page action-row assertions stay deferred until the hosted observer
   lands.
4. Refresh the live-runner fixtures to the current S4 model shape with empty
   action sections instead of preserving the pre-action-section model.
5. Refresh hosted-smoke fixtures to the same current S4 model shape so the
   shared scorecard contract is consistent across live and hosted runners.

### Review Contract

Acceptance criteria:

- `priority_fix_queue`, `top_unresolved_repeats`, `drafted_resolutions`, and
  `already_covered_still_recurring` have model-derived row totals in the
  scorecard counts.
- The four action sections are required model sections, so a producer
  regression that drops the paid action queue cannot false-green as zero rows.
- Default surface caps include action-section caps for `pdf` using each
  section's existing PDF contract limit: 10.
- `build_deflection_full_report_qa_scorecard` fails when a surface observation
  reports fewer displayed action rows than the paid model requires for that
  surface cap.
- `check_deflection_full_report_pdf_export_artifacts.py` extracts action-section
  displayed rows from PDF text and feeds them into the shared scorecard when
  the model has rows for the section.
- Ranked-question PDF row extraction stops at action-section headings so action
  rows cannot hide a missing ranked row.
- The slice does not add buyer hosted-page automation in ATLAS; the buyer
  result page lives in `atlas-portfolio`, so result-page action-row caps remain
  deferred here.
- `tests/test_run_deflection_full_report_qa_live_runner.py` uses current-shape
  fixtures with present action sections, including the valid empty-section
  case.
- `tests/test_smoke_content_ops_deflection_hosted_qa_scorecard.py` uses the
  same current-shape fixture contract for hosted result-page observations.
- Snapshot remains fail-closed by the existing model projection; this slice
  does not make action sections snapshot-safe.

Affected surfaces:

- `extracted_content_pipeline/faq_deflection_report.py`
- `scripts/check_deflection_full_report_pdf_export_artifacts.py`
- scorecard, live-runner, hosted-smoke, and PDF/export validator tests

Risk areas:

- False-green QA if action sections are omitted from rendered PDF text.
- False-positive QA if a section with zero model rows is required to display
  rows.
- Privacy boundary drift if the QA slice tries to project paid action sections
  into Snapshot.

- Reviewer rules triggered: R1, R2, R3, R5, R10, R13, R14.

Max files: 7

### Files touched

- `extracted_content_pipeline/faq_deflection_report.py`
- `plans/PR-Deflection-Action-Section-QA-Scorecard.md`
- `scripts/check_deflection_full_report_pdf_export_artifacts.py`
- `tests/test_content_ops_deflection_report.py`
- `tests/test_run_deflection_full_report_qa_live_runner.py`
- `tests/test_smoke_content_ops_deflection_hosted_qa_scorecard.py`
- `tests/test_smoke_content_ops_deflection_pdf_export_validators.py`

## Mechanism

The shared scorecard requires the four action-section model sections, counts
`items` for each action section, and adds those counts to
`_scorecard_section_total`. Default PDF caps gain action-section entries
matching the model's `pdf_limit`; default result-page caps stay on the
currently observed hosted sections until `atlas-portfolio` ships action-section
row extraction.

The PDF/export validator will split PDF text by the action-section headings and
count visible action questions in each section, using the same question-text
matching helper used by ranked questions and question details. Those
observations feed the existing displayed-row scorecard assertions. Action
headings are model-gated markers, so a model with zero rows for an action
section does not force a heading into every artifact.

## Intentional

- This PR does not change the PDF renderer, report-model producer, delivery
  email, evidence export, or portfolio buyer result page.
- Result-page observations remain limited to the sections the hosted observer
  can currently report. Action-section result-page caps are deferred to the
  `atlas-portfolio` observer slice so this PR does not create an immediate
  false-red gate.
- The checker counts visible action questions, not raw source/evidence fields.
  Complete evidence remains export-only.

## Deferred

- Hosted buyer result-page action-section observation and the corresponding
  result-page action caps remain in `atlas-portfolio`.
- Email summary action-section rendering, if desired, remains a later delivery
  slice.

Parked hardening: none.

## Verification

- Command: python -m pytest tests/test_run_deflection_full_report_qa_live_runner.py::test_live_runner_extracts_pdf_text_from_renderer_bytes_by_default tests/test_run_deflection_full_report_qa_live_runner.py::test_live_runner_fetches_live_json_and_writes_redacted_scorecard -q -- 2 passed.
- Command: python -m pytest tests/test_content_ops_deflection_report.py -q -- 145 passed.
- Command: python -m pytest tests/test_run_deflection_full_report_qa_live_runner.py -q -- 17 passed.
- Command: python -m pytest tests/test_smoke_content_ops_deflection_hosted_qa_scorecard.py -q -- 3 passed.
- Command: python -m pytest tests/test_smoke_content_ops_deflection_pdf_export_validators.py -q -- 18 passed.
- Command: python -m py_compile extracted_content_pipeline/faq_deflection_report.py scripts/check_deflection_full_report_pdf_export_artifacts.py tests/test_content_ops_deflection_report.py tests/test_smoke_content_ops_deflection_hosted_qa_scorecard.py tests/test_smoke_content_ops_deflection_pdf_export_validators.py tests/test_run_deflection_full_report_qa_live_runner.py -- passed.
- Command: bash scripts/validate_extracted_content_pipeline.sh -- passed.
- Command: python extracted/_shared/scripts/forbid_atlas_reasoning_imports.py extracted_content_pipeline -- passed.
- Command: python scripts/audit_extracted_standalone.py --fail-on-debt -- passed.
- Command: bash scripts/check_ascii_python.sh -- passed.
- Command: bash scripts/local_pr_review.sh --current-pr-body-file tmp/pr_body_deflection_action_section_qa_scorecard.md -- passed.

## Estimated diff size

| File | LOC |
|---|---:|
| `extracted_content_pipeline/faq_deflection_report.py` | 57 |
| `plans/PR-Deflection-Action-Section-QA-Scorecard.md` | 167 |
| `scripts/check_deflection_full_report_pdf_export_artifacts.py` | 56 |
| `tests/test_content_ops_deflection_report.py` | 155 |
| `tests/test_run_deflection_full_report_qa_live_runner.py` | 43 |
| `tests/test_smoke_content_ops_deflection_hosted_qa_scorecard.py` | 39 |
| `tests/test_smoke_content_ops_deflection_pdf_export_validators.py` | 144 |
| **Total** | **661** |

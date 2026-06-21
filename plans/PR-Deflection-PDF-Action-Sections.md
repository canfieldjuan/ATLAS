# PR-Deflection-PDF-Action-Sections

## Why this slice exists

Issue #1612 is turning the paid deflection report into an actionable support
ops deliverable. PR #1758/S4A added explicit `pdf_limit` / `result_page_limit`
contract fields for the action-oriented report-model sections, but the PDF
renderer still allowlists only the older section IDs. As a result, the paid PDF
can include ranked questions and details while silently dropping the work queue
sections that tell a help-center owner what to fix first.

Root cause: `atlas_brain.deflection_pdf_renderer._report_model_section_pdf_lines`
admits PDF sections by explicit section ID, and the S4A action-section IDs have
no renderer branch yet. This slice fixes the root at the PDF render boundary by
adding allowlist construction for those action sections instead of widening the
renderer to dump arbitrary section payloads.

## Scope (this PR)

Ownership lane: issue-1612/deflection-full-report-delivery-actionability
Slice phase: Vertical slice

1. Render the S4A action sections in the paid PDF from their existing
   report-model payloads: `priority_fix_queue`, `top_unresolved_repeats`,
   `drafted_resolutions`, and `already_covered_still_recurring`.
2. Use each section's `data.pdf_limit` as the PDF row cap, while keeping the
   result-page limit and broader report-model contract unchanged.
3. Add renderer tests proving the PDF markdown includes only allowlisted
   action-item fields, caps at `pdf_limit`, and does not expose raw evidence,
   source IDs, or representative phrasing payloads.

### Review Contract

Acceptance criteria:

- Paid report-model PDFs render all four action-section IDs when their
  `surfaces` include `pdf`.
- Action-section rendering is allowlist construction: no `top_evidence`,
  `source_ids`, `representative_phrasing`, or raw quote fields are emitted.
- The renderer honors `data.pdf_limit`; it does not reuse the smaller
  `result_page_limit` and does not invent a new backend contract.
- Existing ranked-question, question-detail, and curated-markdown PDF behavior
  remains unchanged.

Affected surfaces:

- `atlas_brain/deflection_pdf_renderer.py`
- `tests/test_deflection_pdf_renderer.py`

Risk areas:

- Privacy/export boundary: action items can carry raw evidence fields in the
  stored model, but the PDF must be a curated surface.
- Report-actionability contract: the PDF should show the top paid queue items,
  not only the web teaser count.

Triggered reviewer rules:

- R1 Requirements match
- R2 Test evidence
- R3 Security/auth/privacy
- R5 Boundary/contract preservation
- R8 Thin-slice scope
- R13 Review-finding class coverage
- R14 Codebase verification

Max files: 3

### Files touched

- `atlas_brain/deflection_pdf_renderer.py`
- `plans/PR-Deflection-PDF-Action-Sections.md`
- `tests/test_deflection_pdf_renderer.py`

## Mechanism

Add explicit action-section branches to `_report_model_section_pdf_lines`.
Those branches call a shared helper that:

- reads `items` and `pdf_limit` from `section.data`;
- emits a short section intro plus an allowlisted table of rank, question,
  status, tickets, estimated support cost, priority score, owner lane, and
  recommended action;
- appends a cap note when more model items exist than the PDF renders;
- ignores raw evidence/source fields even when they are present in the stored
  report model.

This keeps the PDF renderer fail-closed: a new section ID still renders only
after it gets an explicit renderer branch.

## Intentional

- This PR does not change the report-model producer, stored-model normalizer,
  frontend result page, email summary, or complete evidence export. S4A already
  created the contract; this slice consumes it in the PDF only.
- The action-section helper emits compact tables rather than full per-item
  narrative blocks. The PDF should be dense enough to hand to a support lead
  without becoming the complete ticket archive.
- The renderer does not include `top_evidence`, `source_id`, `source_ids`, or
  `representative_phrasing` in action sections. Complete evidence remains in
  the export JSON.

## Deferred

- Full PDF layout polish, richer grouping, and any browser/portfolio result-page
  changes remain later #1612 slices.
- Email attachment/body delivery wiring remains a separate delivery slice.

Parked hardening: none.

## Verification

- Command: python -m pytest tests/test_deflection_pdf_renderer.py -q -- 13 passed.
- Command: python -m py_compile atlas_brain/deflection_pdf_renderer.py tests/test_deflection_pdf_renderer.py -- passed.
- Command: bash scripts/local_pr_review.sh --current-pr-body-file tmp/pr_body_deflection_pdf_action_sections.md -- passed.

## Estimated diff size

| File | LOC |
|---|---:|
| `atlas_brain/deflection_pdf_renderer.py` | 94 |
| `plans/PR-Deflection-PDF-Action-Sections.md` | 124 |
| `tests/test_deflection_pdf_renderer.py` | 83 |
| **Total** | **301** |

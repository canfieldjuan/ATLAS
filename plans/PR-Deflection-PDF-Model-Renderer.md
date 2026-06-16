# PR-Deflection-PDF-Model-Renderer

## Why this slice exists

Epic #1588 is moving paid deflection report surfaces away from one monolithic
Markdown blob and onto a persisted `deflection.v1` section model. #1594 made
the PDF curated/shareable, #1596/#1598 created the model and registry, #1600
persisted the model at the store boundary, and #1603 exposed the model through
a paid API route. The PDF renderer is now the remaining in-repo consumer still
parsing artifact Markdown even when a supported `report_model` is present.

Root cause: the PDF surface's source of truth is still the legacy full-report
Markdown, so the new model contract cannot actually drive the emailed PDF
attachment. This slice fixes the most-upstream safe point in this lane by
making `render_deflection_full_report_pdf(...)` prefer the persisted
`deflection.v1` sections and use Markdown only as a legacy fallback for
historical artifacts that predate the model. It does not touch portfolio UI
rendering because that lives in another checkout/lane.

This exceeds the 400 LOC soft cap because the renderer has to cover every
currently registered PDF section (`support_tax`, `source_file`, `seo_targets`,
`ranked_questions`, `outcome_diagnostics`, and `question_details`) and the same
PR needs the non-happy-path tests that prove model-source selection, legacy
fallback, unknown/export-only skipping, model-row caps, and raw-evidence
exclusion.

## Scope (this PR)

Ownership lane: content-ops/deflection-launch-readiness
Slice phase: Vertical slice

1. Make the deflection PDF renderer prefer artifact `report_model` sections
   when a supported `deflection.v1` model is present.
2. Render PDF-specific Markdown from stored section `data` rather than from
   transient `markdown_lines` or the full artifact `markdown` string.
3. Keep the existing artifact Markdown path as a fallback for legacy paid
   artifacts without a supported stored model.
4. Keep `complete_evidence` export-only for PDF rendering; the PDF points to
   the complete evidence export instead of inlining source IDs/quotes.
5. Preserve PDF readability caps for ranked question rows and question detail
   blocks when rendering from model data.
6. Prove model-source selection, legacy fallback, export-only behavior, and
   model-row capping in `tests/test_deflection_pdf_renderer.py`.
7. Archive the now-merged #1603 plan doc by exact name and refresh
   `plans/INDEX.md`.

### Files touched

- `atlas_brain/deflection_pdf_renderer.py`
- `plans/INDEX.md`
- `plans/PR-Deflection-PDF-Model-Renderer.md`
- `plans/archive/PR-Deflection-Report-Model-API.md`
- `tests/test_deflection_pdf_renderer.py`

### Review Contract

Acceptance criteria:

- [ ] `render_deflection_full_report_pdf(...)` succeeds with a supported
      `report_model` even when artifact `markdown` is absent.
- [ ] When both `report_model` and `markdown` are present, the PDF source is
      the model; stale Markdown-only sentinel content does not enter the
      model-rendered PDF text.
- [ ] Legacy artifacts without a supported model still use the existing
      Markdown renderer path.
- [ ] Model sections are sorted by `priority`, filtered to `pdf` surfaces, and
      unknown/export-only sections are skipped.
- [ ] `complete_evidence` is not rendered inline from the model; the PDF keeps
      a concise complete-export pointer.
- [ ] Ranked rows and question details remain capped from model data with the
      same readability notes used by the curated PDF surface.
- [ ] Delivery worker call shape remains unchanged:
      `render_deflection_full_report_pdf(decoded_artifact)`.
- [ ] #1603's plan is archived by exact name only; no bulk archive sweep moves
      concurrent in-flight plans.

Affected surfaces: Atlas deflection PDF attachment renderer, its focused tests,
and exact-name plan archive housekeeping for #1603.

Risk areas: hiding model/Markdown drift by rendering the wrong source,
breaking historical paid reports, leaking uncapped evidence back into the PDF,
and overfitting to the current fixture order instead of section priorities.

Reviewer rules triggered: R1, R2, R5, R9, R10, R13, R14.

## Mechanism

Use the existing safe stored-model projection from
`extracted_content_pipeline.deflection_report_access.stored_deflection_report_model(...)`
inside the PDF renderer:

1. ask the artifact for a supported `deflection.v1` model;
2. if present, render a PDF-specific Markdown document from sorted model
   sections whose `surfaces` include `pdf`;
3. skip unknown sections and export-only `complete_evidence`;
4. render known section IDs from their stored `data` dictionaries
   (`support_tax`, `source_file`, `seo_targets`, `ranked_questions`,
   `outcome_diagnostics`, and `question_details`);
5. cap model rows before handing the prepared text to the existing FPDF
   Markdown drawing helper;
6. if no supported model is present, keep the existing curated artifact
   Markdown fallback unchanged.

The intermediate string is PDF-specific renderer input, not a return to the
old global Markdown source of truth. It lets this slice reuse the existing FPDF
layout and unicode-safety code while moving the upstream data source to the
persisted model.

## Intentional

- No portfolio hosted-result-page changes. The portfolio checkout is dirty on
  another branch and the web renderer is a separate #1588 consumer.
- No new schema version. This slice consumes the existing persisted
  `deflection.v1` model and skips unsupported models via the current
  store-boundary projection.
- No clickable PDF bookmarks. #1588 keeps clickable navigation optional; the
  existing plain table of contents remains the PDF navigation surface.
- No delivery worker changes. The worker already passes the full paid artifact
  into this renderer; switching the renderer's internal source keeps queue and
  email behavior stable.
- No bulk plan archive sweep. This PR only moves #1603's merged plan by exact
  name; the existing root-plan backlog belongs to its own housekeeping lane.

## Deferred

- Epic #1588 follow-up: move the portfolio hosted paid result page to consume
  the `/report-model` route instead of full artifact Markdown.
- Epic #1588 follow-up: add a direct FPDF section renderer if the current
  Markdown-shaped drawing adapter becomes limiting for layout.
- Epic #1588 optional follow-up: clickable PDF navigation/bookmarks if customer
  usage shows the PDF is a primary navigation surface.

Parked hardening: none.

## Verification

- `python -m pytest tests/test_deflection_pdf_renderer.py -q`
  - Result: 10 passed.
- `python -m pytest tests/test_atlas_content_ops_deflection_delivery.py -q`
  - Result: 15 passed.
- Compile check for the changed renderer and renderer test files.
  - Result: passed.
- `git diff --check`
  - Result: passed.
- `python scripts/sync_pr_plan.py plans/PR-Deflection-PDF-Model-Renderer.md --check`
  - Result: passed.
- Pending before push:
  - push-wrapper local PR review

## Estimated diff size

| File | LOC |
|---|---:|
| `atlas_brain/deflection_pdf_renderer.py` | 454 |
| `plans/INDEX.md` | 1 |
| `plans/PR-Deflection-PDF-Model-Renderer.md` | 159 |
| `plans/archive/PR-Deflection-Report-Model-API.md` | 0 |
| `tests/test_deflection_pdf_renderer.py` | 176 |
| **Total** | **790** |

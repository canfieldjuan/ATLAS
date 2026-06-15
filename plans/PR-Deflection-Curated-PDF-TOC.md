# PR-Deflection-Curated-PDF-TOC

## Why this slice exists

Epic #1588 says the paid deflection PDF should become curated/shareable after
the complete evidence export exists. #1591 added the uncapped
`deflection_evidence.v1` export and #1592 exposed it to paid customers, so the
PDF no longer needs to serve as the complete evidence archive.

Root cause: `render_deflection_full_report_pdf` currently treats the full
artifact Markdown as the PDF model and renders every line into the attachment.
That made large uploads produce unreadable, share-hostile PDFs. This PR fixes
the correct in-scope layer by adding PDF-specific curation and a plain table of
contents while preserving the existing delivery path. The deeper upstream root
is the monolithic Markdown report model; #1588 intentionally defers that to the
structured `deflection.v1` model slice, so this PR does not rewrite the report
model and the PDF renderer together.

This is over the 400 LOC soft cap after review because the slice now includes
the required real-producer integration guard and the heading-shaped evidence
regression. Splitting those tests out would leave the exact producer/consumer
drift and evidence-leak risks that the review identified unguarded.

## Scope (this PR)

Ownership lane: content-ops/deflection-launch-readiness
Slice phase: Product polish

1. Make the deflection PDF renderer produce a curated/shareable PDF from the
   existing paid artifact Markdown.
2. Add a plain, non-clickable table of contents from rendered Markdown headings.
3. Cap ranked table rows and question detail blocks for PDF readability.
4. Collapse `**Complete evidence:**` blocks in the PDF to a short pointer to
   the complete evidence export instead of inlining every source id/quote.
5. Keep the delivery worker interface unchanged: it still calls
   `render_deflection_full_report_pdf(artifact)` and attaches the returned PDF.
6. Keep the hosted result page and evidence-export download route unchanged.
7. Archive merged #1592's plan doc by exact name only and refresh the plan
   index as teardown housekeeping.
8. Extend PDF renderer tests for TOC rendering, top-N curation,
   non-Latin/unicode safety, and the existing missing-Markdown failure.

### Files touched

- `atlas_brain/deflection_pdf_renderer.py`
- `plans/INDEX.md`
- `plans/PR-Deflection-Curated-PDF-TOC.md`
- `plans/archive/PR-Deflection-Evidence-Export-Download.md`
- `tests/test_deflection_pdf_renderer.py`

### Review Contract

Acceptance criteria:

- [ ] PDFs include a plain table of contents generated from `##`/`###`
      headings in the paid artifact Markdown.
- [ ] Ranked table rows and question detail blocks are bounded in the PDF with
      explicit pointer copy to the uncapped export.
- [ ] Full evidence blocks are not rendered inline in the PDF; the PDF instead
      points readers to the complete evidence export for uncapped detail.
- [ ] Heading-shaped text inside source evidence does not end evidence-block
      skipping or leak into the curated PDF.
- [ ] A producer/consumer guard test feeds real
      `build_deflection_report_artifact(...)` Markdown through the curator so
      marker/heading drift fails loudly.
- [ ] Representative non-evidence sections, ranked tables, publishable answer
      copy, and gap guidance still render.
- [ ] Unicode safety remains intact for PDF encoding.
- [ ] Missing artifact Markdown still raises the existing `ValueError`.
- [ ] Delivery worker call shape and attachment behavior stay unchanged.
- [ ] #1592's plan is archived by exact name only; no bulk archive sweep moves
      concurrent in-flight plans.

Affected surfaces: deflection PDF attachment renderer, PDF renderer tests, and
plan archive housekeeping.

Risk areas: accidentally deleting useful report substance, leaving full
evidence inline despite the export, breaking email delivery attachment
generation, and overfitting to one Markdown fixture.

Reviewer rules triggered: R1, R2, R5, R9, R10, R13, R14.

## Mechanism

Keep the renderer Markdown-backed for this slice, but add a PDF-specific
preparation pass before rendering:

1. collect visible headings from the artifact Markdown;
2. render a title section and plain "Table of contents" list before body
   content;
3. cap ranked table rows and question detail blocks at PDF readability limits;
4. stream the Markdown body through the existing renderer while detecting
   `**Complete evidence:**` blocks;
5. replace each complete-evidence block with one concise note that says the
   uncapped source IDs and quotes live in the complete evidence export.
6. keep skipping evidence until the next real section or numbered question
   heading, not arbitrary heading-shaped customer evidence text.

This keeps the existing delivery worker stable and avoids parsing the export
back into the PDF. The upcoming `deflection.v1` model slice can replace this
Markdown preparation pass with model-backed sections.

## Intentional

- No clickable PDF bookmarks/links. #1588 lists clickable navigation as an
  optional later slice; this PR adds the plain TOC only.
- No structured `deflection.v1` model in this PR. The epic explicitly keeps
  model migration as a later strangler slice after the PDF shape is settled.
  This Markdown curation pass is an interim bridge and should be replaced by
  the structured renderer, not allowed to ossify as the final report model.
- No change to delivery-worker email copy, queue claims, or idempotency. This
  slice only changes the PDF bytes the existing renderer returns.
- No hosted result page changes; #1590/#1592 already covered web/dashboard and
  complete export download access.
- Cross-layer caller hints are accounted for: the real non-diff caller is
  `atlas_brain/content_ops_deflection_delivery.py`, covered by the delivery
  worker tests. The extracted blog `_render_markdown` references are a
  same-name unrelated helper, not callers of the PDF renderer.

## Deferred

- Epic #1588 later slice: structured `deflection.v1` paid report model.
- Epic #1588 later slice: section registry for add/remove/reorder behavior.
- Epic #1588 optional slice: clickable PDF navigation/bookmarks if customers
  actually use the PDF heavily.

Parked hardening: none.

## Verification

- `python -m pytest tests/test_deflection_pdf_renderer.py -q`
  - Result: 7 passed.
- `python -m pytest tests/test_atlas_content_ops_deflection_delivery.py -q`
  - Result: 15 passed.
- Compile check for the changed PDF renderer and PDF renderer test files.
  - Result: passed.
- `python scripts/sync_pr_plan.py plans/PR-Deflection-Curated-PDF-TOC.md --check`
  - Result: passed.
- `git diff --check`
  - Result: passed.
- Pending before push:
  - push-wrapper local PR review

## Estimated diff size

| File | LOC |
|---|---:|
| `atlas_brain/deflection_pdf_renderer.py` | 174 |
| `plans/INDEX.md` | 3 |
| `plans/PR-Deflection-Curated-PDF-TOC.md` | 153 |
| `plans/archive/PR-Deflection-Evidence-Export-Download.md` | 0 |
| `tests/test_deflection_pdf_renderer.py` | 173 |
| **Total** | **503** |

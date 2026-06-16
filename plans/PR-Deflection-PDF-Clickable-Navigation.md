# PR-Deflection-PDF-Clickable-Navigation

## Why this slice exists

#1588 split the paid deflection report into surface-specific jobs and left one
optional PDF-specific polish item: clickable navigation/bookmarks if customers
use the PDF heavily. The curated PDF now exists (#1594), the structured report
model and PDF renderer are merged (#1596/#1605), and delivery now sends the
model-backed email wrapper (#1609). That makes PDF navigation the remaining
report-shape polish item that can be added without changing report data or
customer promises.

The current PDF already emits a plain table of contents, but the entries are
static text. In longer curated PDFs, that means the reader still has to scroll
manually. This slice upgrades the existing TOC into internal PDF links and
adds outline/bookmark metadata from the same curated headings, so the PDF is
easier to navigate while keeping the curated/report-model content unchanged.

This branch also archives the just-merged #1609 plan doc by name as normal
post-merge housekeeping. No bulk archive sweep.

## Scope (this PR)

Ownership lane: content-ops/deflection-report-shape
Slice phase: Product polish

1. Render the existing curated TOC entries as internal links to their section
   headings.
2. Add PDF outline/bookmark metadata for the same level-2/level-3 curated
   headings.
3. Preserve current report model selection, Markdown curation, section
   ordering, evidence caps, PDF text, and fallback behavior.
4. Add focused tests for link/outline metadata and no-anchor fallback.
5. Archive the just-merged #1609 plan by name and refresh `plans/INDEX.md`.

### Review Contract

Acceptance criteria:

- TOC entries in generated PDFs create internal link annotations instead of
  static-only text.
- Each rendered level-2/level-3 curated heading registers a destination and PDF
  outline entry; headings removed by PDF curation do not appear in navigation.
- Missing or empty TOC input still renders a valid PDF and does not crash.
- Report content stays curated/shareable: no evidence cap changes, no raw
  evidence/source-ID leakage, no schema/model changes.
- The merged #1609 plan is archived by exact filename only; no concurrent
  in-flight plans are moved.

Affected surfaces:

- `atlas_brain/deflection_pdf_renderer.py` paid deflection PDF rendering.
- `tests/test_deflection_pdf_renderer.py` focused renderer tests.
- Plan archive/index housekeeping for the just-merged #1609 plan.

Risk areas:

- Page-link creation can drift from heading rendering if TOC and renderer
  compute different entries.
- FPDF outline levels can raise if level ordering is invalid.
- Navigation metadata must be tested without asserting fragile full PDF bytes.
- Plan archive housekeeping must not sweep other sessions' active plans.

Triggered reviewer rules:

- R1 requirements match, R2 test evidence, R12 CI/test enrollment, R13 class
  fix, R14 codebase verification.

### Files touched

- `atlas_brain/deflection_pdf_renderer.py`
- `plans/INDEX.md`
- `plans/PR-Deflection-PDF-Clickable-Navigation.md`
- `plans/archive/PR-Deflection-Delivery-Email-Model-Wrapper.md`
- `tests/test_deflection_pdf_renderer.py`

## Mechanism

Reuse the existing curated-heading list as the single source of navigation
truth. Before rendering, build a small TOC destination plan from
`_toc_entries(curated_markdown)`. The intro renders each TOC row with the
corresponding internal link id. The Markdown renderer receives the same plan;
when it renders a matching level-2/level-3 heading, it sets the link target at
the current page/Y position and adds an FPDF outline entry.

The plan is consumed in order so duplicate headings remain deterministic:
the first TOC row links to the first matching rendered heading, the second to
the second, and so on. If no navigation entries exist, the renderer falls back
to the current behavior.

## Intentional

- Keep navigation derived from curated Markdown, not raw Markdown. That avoids
  bookmarks for evidence rows or sections intentionally removed from the PDF.
- Do not redesign PDF layout or add page numbers to the TOC in this slice.
  Internal links and outline entries close the navigation gap without adding a
  fragile pagination pre-pass.
- Do not add a new report-model field for anchors. Navigation is
  renderer-specific and can be derived from already-rendered headings.
- The local caller-hint advisory names other `section_title` methods in B2B
  PDF renderers; those are same-name methods on different classes, not callers
  of `DeflectionReportPDF.section_title`. The real non-diff caller is
  `atlas_brain/content_ops_deflection_delivery.py`, and the public
  `render_deflection_full_report_pdf(...)` signature remains unchanged.

## Deferred

- Page-numbered TOC rows remain deferred. FPDF does not know final page
  destinations without a pre-pass; clickable links/bookmarks provide the
  navigation value without that renderer complexity.
- Direct section-renderer navigation can be added if the Markdown adapter
  becomes limiting. This slice intentionally stays on the current renderer.

Parked hardening: none.

## Verification

- `python -m pytest tests/test_deflection_pdf_renderer.py -q`
  - 12 passed.
- `python -m compileall -q <changed Python files>`
  - Passed.
- `python scripts/sync_pr_plan.py plans/PR-Deflection-PDF-Clickable-Navigation.md --check`
  - Passed.
- Pending before push: push-wrapper local review.

## Estimated diff size

| File | LOC |
|---|---:|
| `atlas_brain/deflection_pdf_renderer.py` | 76 |
| `plans/INDEX.md` | 3 |
| `plans/PR-Deflection-PDF-Clickable-Navigation.md` | 135 |
| `plans/archive/PR-Deflection-Delivery-Email-Model-Wrapper.md` | 0 |
| `tests/test_deflection_pdf_renderer.py` | 100 |
| **Total** | **314** |

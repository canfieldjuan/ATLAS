# PR-Deflection-Uploaded-Search-Contract-Pointer

## Why this slice exists

atlas-portfolio #343 added the deployed uploaded-search smoke, and
atlas-portfolio #344 added cross-reference comments at the two portfolio-side
renderable `TicketFAQItem` validators. Review for #344 noted one remaining
non-blocking drift risk: the third validator, ATLAS
`_deflection_report_full_item`, has no back-pointer to the two portfolio copies.

Root cause: the uploaded-report search admission contract is intentionally
duplicated across backend and portfolio safety gates, but only the portfolio
copies currently advertise the full update set. This slice closes that
documentation gap at the backend copy without changing admission behavior.

## Scope (this PR)

Ownership lane: deflection/uploaded-report-search
Slice phase: Production hardening

1. Add a one-line cross-reference comment next to `_deflection_report_full_item`.
2. Name both portfolio copies: `isRenderableItem` in the Atlas client parser and
   `summarizeRenderableItem` in the uploaded-search smoke.
3. Leave validator logic and tests unchanged except for verification that the
   existing renderable-item test still passes.

### Review Contract

- Acceptance criteria:
  - `_deflection_report_full_item` explicitly points future contract edits at the
    two portfolio-side copies.
  - No runtime logic changes are introduced.
- Affected surfaces: uploaded-report search backend admission comments only.
- Risk areas: accidental validator behavior changes, stale or inaccurate path
  references.
- Reviewer rules triggered: R1, R10, R14.

### Files touched

- `extracted_content_pipeline/api/control_surfaces.py`
- `plans/PR-Deflection-Uploaded-Search-Contract-Pointer.md`

## Mechanism

Add a short comment immediately above `_deflection_report_full_item` naming:

- atlas-portfolio `web/src/lib/atlas-deflection-client.ts` `isRenderableItem`;
- atlas-portfolio `web/scripts/smoke-deflection-uploaded-search.mjs`
  `summarizeRenderableItem`.

The helper body is untouched.

## Intentional

- Comment-only. Sharing a validator between ATLAS and atlas-portfolio would be a
  cross-repo package/design change; this slice only closes the review-identified
  drift reminder.

## Deferred

- A shared cross-repo renderable-item validator remains deferred unless this
  contract starts changing frequently enough to justify the coupling.

Parked hardening: none.

## Verification

- pytest tests/test_extracted_content_deflection_submit.py::test_deflection_report_search_only_returns_portfolio_renderable_items -q -- passed, 1 test.
- bash scripts/validate_extracted_content_pipeline.sh -- passed.
- python extracted/_shared/scripts/forbid_atlas_reasoning_imports.py extracted_content_pipeline -- passed.
- python scripts/audit_extracted_standalone.py --fail-on-debt -- passed.
- bash scripts/check_ascii_python.sh -- passed.

## Estimated diff size

| File | LOC |
|---|---:|
| `extracted_content_pipeline/api/control_surfaces.py` | 2 |
| `plans/PR-Deflection-Uploaded-Search-Contract-Pointer.md` | 80 |
| **Total** | **82** |

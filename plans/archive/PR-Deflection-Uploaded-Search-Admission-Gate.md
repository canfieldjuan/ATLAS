# PR-Deflection-Uploaded-Search-Admission-Gate

## Why this slice exists

#1747 added the paid uploaded-report search endpoint that the portfolio can call
after a CSV-backed report is unlocked. The reviewer found a non-blocking contract
gap: ATLAS admitted some "full" FAQ rows with looser Python checks than the
portfolio `isRenderableItem` parser, so the backend could return a search result
that the UI would reject instead of rendering.

Root cause: `_deflection_report_full_item` checked for broad sequences and
coerced-ish counts instead of the exact render-facing `TicketFAQItem` fields the
portfolio validates before showing an uploaded search match.

## Scope (this PR)

Ownership lane: deflection/uploaded-report-search
Slice phase: Production hardening

1. Tighten uploaded-report search admission so returned `item` payloads must have
   the portfolio-rendered strings, numeric fields, string arrays, and
   `term_mappings` object strings.
2. Add focused route coverage proving malformed full-looking rows are skipped
   while a valid stored paid artifact item still returns.

### Review Contract

- Acceptance criteria:
  - `POST /content-ops/deflection-reports/{request_id}/search` continues to
    return a full paid uploaded report item for valid stored artifacts.
  - Rows missing portfolio-rendered fields, with non-number `ticket_count` or
    `opportunity_score`, with non-string array elements, or with malformed
    `term_mappings` are skipped rather than returned.
  - The change remains scoped to search-result admission and tests; no producer,
    storage, auth, payment, or portfolio changes ride along.
- Affected surfaces:
  - `extracted_content_pipeline/api/control_surfaces.py`
  - `tests/test_extracted_content_deflection_submit.py`
- Risk areas:
  - Accidentally rejecting valid producer output.
  - Reintroducing compact search rows that the portfolio cannot render.
- Reviewer rules triggered: R1, R2, R10, R14.

### Files touched

- `extracted_content_pipeline/api/control_surfaces.py`
- `plans/PR-Deflection-Uploaded-Search-Admission-Gate.md`
- `tests/test_extracted_content_deflection_submit.py`

## Mechanism

`_deflection_report_full_item` now mirrors the portfolio server parser's
rendered-field contract before search scoring can return an item:

- required rendered text fields must already be strings;
- `ticket_count` and `opportunity_score` must be JSON-number-like Python
  `int`/`float` values, not bools or digit strings;
- `steps`, `action_items`, `source_ids`, and `source_labels` must be lists of
  strings;
- `term_mappings` must be a list whose entries expose string
  `customer_term`, `documentation_term`, and `suggestion` fields.

Malformed rows still fail closed by returning `None` from the helper, which keeps
the existing search loop behavior: skip bad rows, score valid rows, and return the
best matches from the stored paid artifact only.

## Intentional

- The helper intentionally remains local to uploaded-report search instead of
  importing or generating a shared TypeScript/Python schema in this small slice.
- Empty strings and empty arrays remain accepted when their types match the
  portfolio parser; search scoring still controls whether a row matches the
  user's query.
- The search endpoint still skips malformed stored rows instead of failing the
  whole request, preserving the #1747 behavior for mixed artifacts.

## Deferred

None.

Parked hardening: none.

## Verification

- pytest tests/test_extracted_content_deflection_submit.py::test_deflection_report_search_returns_full_paid_uploaded_item tests/test_extracted_content_deflection_submit.py::test_deflection_report_search_skips_compact_and_malformed_items tests/test_extracted_content_deflection_submit.py::test_deflection_report_search_only_returns_portfolio_renderable_items -q -- 3 passed.
- pytest tests/test_extracted_content_deflection_submit.py -q -- 75 passed.
- python -m py_compile extracted_content_pipeline/api/control_surfaces.py tests/test_extracted_content_deflection_submit.py -- passed.
- bash scripts/local_pr_review.sh --current-pr-body-file tmp/pr-body-deflection-uploaded-search-admission-gate.md -- passed.

## Estimated diff size

| File | LOC |
|---|---:|
| `extracted_content_pipeline/api/control_surfaces.py` | 40 |
| `plans/PR-Deflection-Uploaded-Search-Admission-Gate.md` | 97 |
| `tests/test_extracted_content_deflection_submit.py` | 41 |
| **Total** | **178** |

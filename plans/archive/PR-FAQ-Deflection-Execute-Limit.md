# PR-FAQ-Deflection-Execute-Limit

## Why this slice exists

`faq_deflection_report` now runs through the same source-material FAQ generation
path as `faq_markdown`, but the sync execute admission guard only recognizes
`faq_markdown`. That leaves the new customer-facing report output able to bypass
the configured inline source row cap.

This slice closes that survivability gap at the shared route guard instead of
adding per-service patchwork.

## Scope (this PR)

Ownership lane: content-ops/deflection-report

Slice phase: Production hardening

1. Apply the existing FAQ sync execute source-material row limit to both
   `faq_markdown` and `faq_deflection_report`.
2. Add a focused negative route test proving oversized `faq_deflection_report`
   source material fails before service execution.
3. Count the expanded source rows the execute path will consume, including FAQ
   output bundle `items`.

### Files touched

| File | Purpose |
|---|---|
| `extracted_content_pipeline/api/control_surfaces.py` | Extends the FAQ execute row-limit guard to cover the deflection report output. |
| `tests/test_extracted_content_control_surface_api.py` | Adds the regression test for the deflection report guard branch. |
| `plans/PR-FAQ-Deflection-Execute-Limit.md` | Documents this slice contract. |

## Mechanism

The route-level helper that enforces `faq_execute_max_source_material_rows`
now checks a single FAQ-limited output set. If the requested outputs include
either FAQ Markdown or the deflection report, it counts rows in `inputs.source_material`
through the same `source_material_to_source_rows(...)` adapter the execute path
uses. That keeps generated FAQ output bundles from being counted as one top-level
object when execution will expand their `items` into multiple rows.

The guard returns the existing 413 envelope when the configured cap is exceeded.

## Intentional

- This reuses the existing 413 reason and payload so operators do not get a new
  error contract for the same upload-size policy.
- No runtime service behavior changes; the guard runs before service dispatch.
- The row counter now reuses the shared source-row adapter instead of keeping a
  second partial row-count implementation in the route.

## Deferred

- Parked hardening: none.
- Future robust-testing slice: larger `faq_deflection_report` route generation
  throughput and latency proof once the report output is exercised under the
  same scale harness as `faq_markdown`.

## Verification

- Command: python -m pytest tests/test_extracted_content_control_surface_api.py -q -k "faq_source_material or source_material_bundle_1000 or non_source_material_arrays or nested_source_material"
  - Result: 7 passed, 113 deselected.
- Command: python -m py_compile extracted_content_pipeline/api/control_surfaces.py tests/test_extracted_content_control_surface_api.py
  - Result: passed.
- Command: python scripts/audit_plan_doc.py plans/PR-FAQ-Deflection-Execute-Limit.md
  - Result: passed.
- Command: python scripts/audit_plan_code_consistency.py plans/PR-FAQ-Deflection-Execute-Limit.md
  - Result: passed.
- Command: git diff --check
  - Result: passed.
- Command: bash scripts/local_pr_review.sh --current-pr-body-file /home/juan-canfield/Desktop/atlas-pr-bodies/faq-deflection-execute-limit.md
  - Result: passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Route guard | 20 |
| Tests | 80 |
| Plan doc | 82 |
| **Total** | **182** |

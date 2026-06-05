# PR: Content Ops Competitive Sales Brief Handoff

## Why this slice exists

PR #1253 made competitive/displacement evidence usable for landing pages and
blog posts. PR #1258 and #1260 then made canonical B2B displacement rows
selectable from the operator UI. The remaining marketer-output gap is that the
same competitive evidence still cannot produce a sales-facing generated asset:
`sales_brief` reads repository opportunities and ignores the source material
package the competitive input provider already prepares.

This slice is the thinnest competitive-specific output follow-up. It does not
invent a new asset type; it reuses the existing `SalesBriefGenerationService`
and its `displacement` brief shape, but lets the Content Ops executor hand it
packaged source-material opportunities for competitive runs.

## Scope (this PR)

Ownership lane: content-ops/marketer-reviews-as-input
Slice phase: Vertical slice

1. Let `SalesBriefGenerationService.generate(...)` accept optional
   `source_material` and build normalized campaign opportunities from it before
   falling back to the repository.
2. Thread `request.inputs.source_material` through the sales-brief dispatcher.
3. Add `sales_brief` to the competitive input package defaults and set
   `brief_type=displacement`.
4. Add focused tests proving competitive source material reaches the sales brief
   path and defaults to the displacement brief type.

### Files touched

- `plans/PR-Content-Ops-Competitive-Sales-Brief-Handoff.md`
- `atlas_brain/_content_ops_input_provider.py`
- `extracted_content_pipeline/content_ops_execution.py`
- `extracted_content_pipeline/sales_brief_generation.py`
- `tests/test_atlas_content_ops_review_input_provider.py`
- `tests/test_extracted_content_ops_execution.py`
- `tests/test_extracted_sales_brief_generation.py`

## Mechanism

`SalesBriefGenerationService.generate` will gain a `source_material` keyword.
When it is supplied, the service will use the existing source-row adapter to
normalize the material into campaign opportunities and skip the repository read.
When it is omitted, the current repository-backed behavior stays unchanged.

The Content Ops sales-brief dispatcher will pass `request.inputs.source_material`
to the service. The competitive input package already stores the filtered
competitive opportunities under that key, so adding `sales_brief` to the package
outputs plus `brief_type=displacement` gives the existing generated asset a real
competitive path.

## Intentional

- No new generated asset table, router, UI review screen, or output id. Sales
  briefs already exist and already support `displacement` copy framing.
- No live B2B SQL change. The canonical B2B displacement loader landed in
  #1258 and the UI selector landed in #1260.
- Source-material mode is opt-in: repository-backed sales briefs still use the
  current `read_campaign_opportunities` path when no source material is present.

## Deferred

- Social posts, ad copy, and stat/quote card outputs remain future slices.
- Product packaging/pricing for the marketer competitive offer remains
  deferred.

## Parked hardening

None.

## Verification

- Passed: pytest tests/test_extracted_sales_brief_generation.py tests/test_extracted_content_ops_execution.py tests/test_atlas_content_ops_review_input_provider.py -q (107 passed)
- Passed: python -m py_compile extracted_content_pipeline/sales_brief_generation.py extracted_content_pipeline/content_ops_execution.py atlas_brain/_content_ops_input_provider.py
- Passed: bash scripts/validate_extracted_content_pipeline.sh
- Passed: python extracted/_shared/scripts/forbid_atlas_reasoning_imports.py extracted_content_pipeline
- Passed: python scripts/audit_extracted_standalone.py --fail-on-debt
- Passed: bash scripts/check_ascii_python.sh
- Passed: python scripts/audit_extracted_pipeline_ci_enrollment.py --atlas-brain-tests-from origin/main (OK: 144 matching tests are enrolled.)
- Passed: bash scripts/run_extracted_pipeline_checks.sh (2957 passed, 10 skipped, 1 warning)
- Passed: bash scripts/local_pr_review.sh --current-pr-body-file /tmp/content-ops-competitive-sales-brief-handoff-pr-body.md

## Estimated diff size

Actual: 7 files, +242 / -26.

| Area | Estimated LOC |
|---|---:|
| Sales-brief source-material handoff | ~75 |
| Executor + provider defaults | ~25 |
| Focused tests | ~140 |
| Plan doc | ~90 |
| **Total** | **~330** |

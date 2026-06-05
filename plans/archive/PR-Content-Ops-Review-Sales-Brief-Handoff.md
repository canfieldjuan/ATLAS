# PR: Content Ops Review Sales Brief Handoff

## Why this slice exists

PR #1247 made customer review/source evidence usable as input for landing pages
and blog posts. PR #1261 then added the missing service-side source-material
path for `sales_brief`, but only competitive packages were enrolled as a
sales-brief output. That leaves the review marketer path uneven: the same
review-grounded source rows can produce public landing/blog assets, but not the
internal sales-facing brief that a rep can use from those review themes.

This slice reuses the existing `sales_brief` output and the source-material
handoff that already landed. It only enrolls review-source packages in that
output set and proves the review evidence reaches the sales-brief dispatcher.

## Scope (this PR)

Ownership lane: content-ops/marketer-reviews-as-input
Slice phase: Vertical slice

1. Add `sales_brief` to the review-source input package output defaults.
2. Set review-source packages to use `brief_type=discovery`, which matches a
   review-grounded sales enablement brief better than renewal or displacement.
3. Update focused provider/execution tests so review source material reaches
   landing, blog, and sales-brief services from the same package.

### Files touched

- `plans/PR-Content-Ops-Review-Sales-Brief-Handoff.md`
- `atlas_brain/_content_ops_input_provider.py`
- `tests/test_atlas_content_ops_review_input_provider.py`

## Mechanism

The review input provider already normalizes raw review/source rows into flat
campaign opportunities and stores them under both `source_material` and
`review_source_material`. PR #1261 made the sales-brief executor pass
`request.inputs.source_material` into `SalesBriefGenerationService.generate`.

This PR therefore only changes the review package defaults:

- `_REVIEW_CAMPAIGN_OUTPUTS` becomes `("landing_page", "blog_post", "sales_brief")`.
- The review input payload gains `brief_type: "discovery"` so the existing plan
  builder emits that as the sales-brief `default_brief_type`.

## Intentional

- No generator, prompt, router, DB, or UI review-screen change. The sales-brief
  generator already accepts source material after #1261.
- No competitive-path change. Competitive packages already use
  `brief_type=displacement`.
- Review sales briefs use `discovery`, not `pre_call`, because the source rows
  are third-party review themes rather than account-specific meeting notes.

## Deferred

- Social posts, ad copy, and stat/quote card outputs remain future slices.
- Product packaging/pricing for the review/competitive marketer offer remains
  deferred.

## Parked hardening

None.

## Verification

- Passed: pytest tests/test_atlas_content_ops_review_input_provider.py -q (22 passed)
- Passed: python -m py_compile atlas_brain/_content_ops_input_provider.py
- Passed: python scripts/audit_extracted_pipeline_ci_enrollment.py --atlas-brain-tests-from origin/main (OK: 144 matching tests are enrolled.)
- Passed: git diff --check
- Passed: bash scripts/run_extracted_pipeline_checks.sh (2957 passed, 10 skipped, 1 warning)
- Passed: bash scripts/local_pr_review.sh --current-pr-body-file /tmp/content-ops-review-sales-brief-handoff-pr-body.md

## Estimated diff size

Actual: 3 files, +99 / -7.

| Area | Estimated LOC |
|---|---:|
| Review provider defaults | ~5 |
| Focused tests | ~60 |
| Plan doc | ~85 |
| **Total** | **~150** |

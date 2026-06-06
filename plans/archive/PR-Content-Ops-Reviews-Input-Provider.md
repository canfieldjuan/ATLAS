# PR: Content Ops Reviews Input Provider

## Why this slice exists

PR #1237 landed the scoping plan for marketer-facing review inputs. The
generation engine already understands review-shaped source rows through
`campaign_source_adapters.py`, but the host Content Ops input provider still
routes only support-ticket source material. This vertical slice makes the
existing New Run path explicit about source type so review rows can drive the
already-built landing-page and blog generators without creating a second
pipeline. The diff is over the preferred 400 LOC budget because the slice is
only useful if the backend provider, operator-facing selector, and required
negative fixtures land together; splitting any one of those would leave either
an unselectable backend path or an untested source-type gate.

## Scope (this PR)

Ownership lane: content-ops/marketer-reviews-as-input
Slice phase: Vertical slice

1. Add explicit source-type selection to the Atlas host input provider.
2. Add a review input package path that reuses existing source-row adapters.
3. Add a New Run source selector that writes the source type into inputs JSON.
4. Add focused backend and frontend tests for review routing and support-ticket
   preservation.
5. Enroll the new Atlas-host test file in the dedicated input-provider workflow.
6. Review-fix pass: seed request-level review type into adapter defaults,
   carry review evidence into landing/blog generation contexts, and enroll the
   frontend selector test in the explicit intel-ui CI workflow.

### Files touched

- `plans/PR-Content-Ops-Reviews-Input-Provider.md`
- `.github/workflows/atlas_content_ops_input_provider_checks.yml`
- `.github/workflows/atlas_intel_ui_checks.yml`
- `scripts/run_extracted_pipeline_checks.sh`
- `atlas_brain/_content_ops_input_provider.py`
- `extracted_content_pipeline/content_ops_execution.py`
- `extracted_content_pipeline/landing_page_input_contract.py`
- `atlas-intel-ui/src/pages/ContentOpsNewRun.tsx`
- `atlas-intel-ui/src/pages/contentOpsSourceMode.ts`
- `atlas-intel-ui/package.json`
- `atlas-intel-ui/scripts/content-ops-review-source-selection.test.mjs`
- `tests/test_atlas_content_ops_review_input_provider.py`

## Mechanism

The host provider reads `inputs.source_type` with `inputs.source_material_type`
as a compatibility alias. Missing source type preserves the current
support-ticket autodetect behavior. Explicit `support_ticket` keeps the existing
support-ticket provider. Explicit `reviews` / `review` routes source material
through `source_material_to_source_rows` and
`source_rows_to_campaign_opportunities`, accepts only review or complaint rows,
and returns a Content Ops input package for `landing_page` and `blog_post`.
Unsupported explicit source types fail closed with a noop package and warning.

The New Run UI adds a compact source selector beside the raw inputs controls.
The support-ticket option removes the explicit key for backward compatibility;
the review option writes `"source_type": "reviews"` and hides saved FAQ source
selection because saved FAQ drafts are support-ticket-specific.

Review rows are also copied to `review_source_material` so the executor can
place the evidence in landing-page campaign context and blog `data_context`.
This keeps the operator-selected review source visible to the generators rather
than only to the input-provider diagnostics.

## Intentional

- No new endpoint: preview, plan, and execute already call the host input
  provider.
- No new review parser: reuse `campaign_source_adapters.py`.
- No image-provider work in this slice; PR #1244 is separate.
- Missing source type remains backward-compatible support-ticket autodetection.

## Deferred

- Competitive/displacement-as-input remains parked for a later marketer slice.
- Social/ad/stat-card outputs remain separate output work.
- Image generation and image attachment remain in the image-provider lane.

## Parked hardening

None.

## Verification

- pytest for review input provider plus support-ticket provider regressions --
  35 passed, 1 warning.
- Focused review-input pytest (`tests/test_atlas_content_ops_review_input_provider.py`) -- 11
  passed.
- Frontend review source-selection npm script -- 4 passed.
- `npm run lint` -- passed.
- `npm run build` -- passed.
- py_compile for changed Python files -- passed.
- Extracted CI enrollment audit
  -- OK: 143 matching tests are enrolled.
- Full extracted pipeline mirror (`scripts/run_extracted_pipeline_checks.sh`) -- 2924 passed, 10
  skipped, 1 warning; all extracted content pipeline checks completed.
- `bash scripts/local_pr_review.sh --current-pr-body-file /tmp/content-ops-reviews-input-provider-pr-body.md`
  -- local PR review passed.

## Estimated diff size

Estimated: 982 LOC (12 files, +955 / -27 actual). This intentionally exceeds the preferred 400 LOC budget to
keep the review provider, New Run selector, same-PR frontend enrollment, and
source-type negative fixtures in one vertical slice.

| Area | Estimated LOC |
|---|---:|
| Backend provider + executor context | ~278 |
| Backend tests | ~362 |
| Frontend selector helper and test enrollment | ~153 |
| Workflow and runner enrollment | ~8 |
| Plan doc | ~115 |
| **Total** | **982** |

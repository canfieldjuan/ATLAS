# PR: Content Ops Marketer Evidence Preset

## Why this slice exists

PR #1261 and PR #1263 completed output parity for review and competitive
source-material packages: both can now hand their evidence to `landing_page`,
`blog_post`, and `sales_brief`. The operator can still only select those assets
manually or use broader presets that include unrelated outputs, so the product
bundle is not exposed as a named control-surface option.

This slice productizes the completed marketer evidence path without adding a
new generator. It gives the UI/API a stable preset for the landing page, blog
post, and sales brief bundle that review and competitive source-material runs
now support end to end.

## Scope (this PR)

Ownership lane: content-ops/marketer-reviews-as-input
Slice phase: Product polish

1. Add a `marketer_evidence_bundle` control-surface preset for
   `landing_page`, `blog_post`, and `sales_brief`.
2. Prove the preset resolves through the pure preview path and is exposed by
   the control-surfaces API catalog.
3. Keep docs and the UI catalog fixture aligned with the producer output.

### Files touched

- `plans/PR-Content-Ops-Marketer-Evidence-Preset.md`
- `extracted_content_pipeline/control_surfaces.py`
- `extracted_content_pipeline/docs/control_surface_preview_api.md`
- `tests/test_extracted_content_control_surfaces.py`
- `tests/test_extracted_content_control_surface_api.py`
- `atlas-intel-ui/src/api/__fixtures__/contentOps/catalog.json`

## Mechanism

`PRESETS` is the single source for named output bundles. Adding
`marketer_evidence_bundle` there makes the existing preview, plan, API catalog,
and UI preset rendering paths pick up the bundle automatically.

The preset intentionally references only already-implemented outputs:

```python
outputs=("landing_page", "blog_post", "sales_brief")
```

The focused tests cover both the pure extracted resolver and the API catalog
serialization so a future preset rename, missing catalog exposure, or wrong
output list fails locally.

## Intentional

- No new generated asset type. Social posts, ad copy, and stat/quote cards are
  still separate future outputs.
- No UI component change. `ContentOpsNewRun` already renders
  `catalog.presets`, so changing the producer catalog is the narrowest path.
- No provider-default change. Review and competitive input packages already
  default to this output set after #1261 and #1263; this PR names the same set
  for manual/API control-surface selection.

## Deferred

- Social posts, ad copy, and stat/quote card outputs remain future slices.
- Pricing/packaging copy outside the control-surface preset remains deferred to
  the marketer offer packaging slice.

## Parked hardening

None.

## Verification

- Passed: pytest tests/test_extracted_content_control_surfaces.py tests/test_extracted_content_control_surface_api.py -q (160 passed, 1 skipped)
- Passed: python -m py_compile extracted_content_pipeline/control_surfaces.py
- Passed: git diff --check
- Passed: npm run lint, from atlas-intel-ui
- Passed: npm run build, from atlas-intel-ui
- Passed: bash scripts/validate_extracted_content_pipeline.sh
- Passed: python extracted/_shared/scripts/forbid_atlas_reasoning_imports.py extracted_content_pipeline
- Passed: python scripts/audit_extracted_standalone.py --fail-on-debt
- Passed: bash scripts/check_ascii_python.sh
- Passed: python scripts/audit_extracted_pipeline_ci_enrollment.py --atlas-brain-tests-from origin/main (OK: 144 matching tests are enrolled.)
- Passed: bash scripts/run_extracted_pipeline_checks.sh (2958 passed, 10 skipped, 1 warning)
- Passed: bash scripts/local_pr_review.sh --current-pr-body-file /tmp/content-ops-marketer-evidence-preset-pr-body.md

## Estimated diff size

Actual: 6 files, +146 / -0.

| Area | Estimated LOC |
|---|---:|
| Preset catalog + docs/fixture | ~35 |
| Focused tests | ~25 |
| Plan doc | ~80 |
| **Total** | **~140** |

# PR: Content Ops Marketer Social Post Output

## Why this slice exists

PR #1264 productized the completed marketer evidence bundle for landing pages,
blog posts, and sales briefs. The same plan deliberately deferred the next
marketer asset types: social posts, ad copy, and stat/quote cards. None of
those outputs exists in the Content Ops catalog today, so operators cannot turn
review or competitive source material into short-form share copy through the
same execution path.

This slice adds the first new marketer asset type as a deterministic vertical
slice: `social_post` runs from uploaded/source-material evidence and returns
bounded, evidence-backed post drafts. It does not add LLM prompting, storage, or
UI controls beyond the existing catalog-driven output list.

This PR exceeds the 400 LOC soft cap because the first executable output type
cannot be split cleanly: the service, manifest entry, catalog exposure,
generation plan, executor dispatch, host wiring, UI fixture, reasoning no-op
policy, and tests all have to land together or the output is either invisible or
not runnable.

## Scope (this PR)

Ownership lane: content-ops/marketer-reviews-as-input
Slice phase: Vertical slice

1. Add a pure `SocialPostGenerationService` that turns source material into
   bounded social post drafts with source ids and non-fatal warnings.
2. Register `social_post` in the control-surface catalog, generation plan, and
   executor service bundle.
3. Wire the always-safe deterministic service into the host execution-service
   factory.
4. Add focused tests for the service, preview/plan mapping, executor dispatch,
   and API catalog readiness.

### Files touched

- `plans/PR-Content-Ops-Marketer-Social-Post-Output.md`
- `.github/workflows/atlas_content_ops_input_provider_checks.yml`
- `extracted_content_pipeline/social_post_generation.py`
- `extracted_content_pipeline/manifest.json`
- `extracted_content_pipeline/control_surfaces.py`
- `extracted_content_pipeline/docs/control_surface_preview_api.md`
- `extracted_content_pipeline/generation_plan.py`
- `extracted_content_pipeline/content_ops_execution.py`
- `extracted_content_pipeline/reasoning_policy.py`
- `atlas_brain/_content_ops_services.py`
- `tests/test_extracted_social_post_generation.py`
- `tests/test_extracted_content_control_surfaces.py`
- `tests/test_extracted_content_generation_plan.py`
- `tests/test_extracted_content_ops_execution.py`
- `tests/test_extracted_content_control_surface_api.py`
- `tests/test_extracted_content_reasoning_policy.py`
- `tests/test_atlas_content_ops_execution_services.py`
- `atlas-intel-ui/src/api/__fixtures__/contentOps/catalog.json`

## Mechanism

`SocialPostGenerationService.generate(...)` accepts `source_material`, normalizes
rows through the existing source-row-to-campaign-opportunity adapter, then emits
one short post per usable opportunity up to `limit`.

The output is intentionally deterministic:

- `source_material` is required by the control surface.
- `source_max_text_chars` controls the adapter text cap, matching
  `signal_extraction`.
- The service returns `{generated, posts, warnings, target_mode}` and does not
  persist or call an LLM.

The host factory wires one reusable deterministic service, so
`GET /content-ops/control-surfaces` reports `social_post` as executable when
normal execution services are configured.

## Intentional

- No LLM social-copy prompt yet. The first slice proves the asset path and keeps
  copy evidence-bounded; richer channel/style variants can layer on later.
- No DB migration or generated-asset review screen. The executor response is
  enough to prove the new output can run from source material.
- No default review/competitive package enrollment yet. This PR exposes the
  output and proves execution; the follow-up can add it to package defaults once
  the reviewer has accepted the output shape.

## Deferred

- Add `social_post` to review and competitive input-package defaults in the next
  marketer-output handoff slice.
- Ad copy and stat/quote card outputs remain future slices.
- LLM/style/channel variants for social posts remain future product polish.

## Parked hardening

None.

## Verification

- Passed: pytest tests/test_extracted_social_post_generation.py -q (4 passed)
- Passed: pytest tests/test_extracted_content_reasoning_policy.py tests/test_extracted_social_post_generation.py tests/test_extracted_content_control_surfaces.py tests/test_extracted_content_generation_plan.py tests/test_extracted_content_ops_execution.py tests/test_extracted_content_control_surface_api.py tests/test_atlas_content_ops_execution_services.py -q (308 passed, 1 skipped)
- Passed: python -m py_compile extracted_content_pipeline/social_post_generation.py extracted_content_pipeline/control_surfaces.py extracted_content_pipeline/generation_plan.py extracted_content_pipeline/content_ops_execution.py extracted_content_pipeline/reasoning_policy.py atlas_brain/_content_ops_services.py
- Passed: git diff --check
- Passed: npm run lint, from atlas-intel-ui
- Passed: npm run build, from atlas-intel-ui
- Passed: bash scripts/validate_extracted_content_pipeline.sh
- Passed: python extracted/_shared/scripts/forbid_atlas_reasoning_imports.py extracted_content_pipeline
- Passed: python scripts/audit_extracted_standalone.py --fail-on-debt
- Passed: bash scripts/check_ascii_python.sh
- Passed: python scripts/audit_extracted_pipeline_ci_enrollment.py --atlas-brain-tests-from origin/main (OK: 144 matching tests are enrolled.)
- Passed: bash scripts/run_extracted_pipeline_checks.sh (2964 passed, 10 skipped, 1 warning)
- Passed: bash scripts/local_pr_review.sh --current-pr-body-file /tmp/content-ops-marketer-social-post-output-pr-body.md

## Estimated diff size

Actual: 18 files, +715 / -5. This is over the soft cap for the indivisible
first-output wiring reason described in **Why this slice exists**.

| Area | Estimated LOC |
|---|---:|
| Social-post service | 224 |
| Catalog/plan/executor/host/reasoning/workflow wiring | ~74 |
| Focused tests + fixture/docs alignment | ~385 |
| Plan doc | 122 |
| **Total** | **~705** |

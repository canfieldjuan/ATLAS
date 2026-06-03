# PR: Content Ops Marketer Ad Copy Output

## Why this slice exists

The marketer reviews-as-input lane has now productized one new short-form
output, `social_post`, end to end through execution, package defaults,
persistence, and generated-asset review. The same lane still has two deferred
marketer asset families: ad copy and stat/quote cards. Operators can turn review
or competitive source material into landing pages, blog posts, sales briefs,
and social posts, but not short paid-media copy.

This slice adds the first ad-copy vertical slice as a deterministic
source-evidence generator. It mirrors the first `social_post` output slice:
register an executable `ad_copy` output, generate bounded evidence-backed ad
drafts from `source_material`, wire host execution, and prove preview/plan/run
behavior. It does not touch #1268's plan-only output-variations lane.

This PR may exceed the 400 LOC soft cap for the same indivisible first-output
reason as `social_post`: the service, manifest entry, catalog exposure,
generation plan, executor dispatch, host wiring, reasoning no-op policy, UI
fixture, and tests need to land together or `ad_copy` is either invisible or not
runnable.

## Scope (this PR)

Ownership lane: content-ops/marketer-reviews-as-input
Slice phase: Vertical slice

1. Add a package-owned deterministic `AdCopyGenerationService` that turns
   source material into bounded ad-copy drafts with channel/format metadata and
   non-fatal source-material warnings.
2. Register `ad_copy` in the control-surface catalog, generation plan, executor
   service bundle, host service factory, and no-op reasoning policy.
3. Keep ad-copy generation source-evidence-only: no LLM prompt, persistence,
   generated-asset review API/UI branch, or package-default enrollment in this
   slice.
4. Add focused tests for service output/warnings/limits, preview/plan mapping,
   executor dispatch, host factory wiring, and catalog readiness.

### Files touched

- `plans/PR-Content-Ops-Marketer-Ad-Copy-Output.md`
- `extracted_content_pipeline/ad_copy_generation.py`
- `extracted_content_pipeline/manifest.json`
- `extracted_content_pipeline/control_surfaces.py`
- `extracted_content_pipeline/docs/control_surface_preview_api.md`
- `extracted_content_pipeline/generation_plan.py`
- `extracted_content_pipeline/content_ops_execution.py`
- `extracted_content_pipeline/reasoning_policy.py`
- `atlas_brain/_content_ops_services.py`
- `tests/test_extracted_ad_copy_generation.py`
- `tests/test_extracted_content_control_surfaces.py`
- `tests/test_extracted_content_generation_plan.py`
- `tests/test_extracted_content_ops_execution.py`
- `tests/test_extracted_content_control_surface_api.py`
- `tests/test_extracted_content_reasoning_policy.py`
- `tests/test_atlas_content_ops_execution_services.py`
- `atlas-intel-ui/src/api/__fixtures__/contentOps/catalog.json`

## Mechanism

`AdCopyGenerationService.generate(...)` accepts `source_material`, normalizes
each row through the existing `source_row_to_campaign_opportunity(...)` adapter,
and emits one ad draft per usable opportunity up to `limit`. Each draft carries:

```python
{
    "id": source_id,
    "format": "search_headline",
    "headline": "...",
    "primary_text": "...",
    "cta": "See the proof",
    "source_id": source_id,
    "source_type": "review",
    "company_name": "...",
    "vendor_name": "...",
    "pain_points": [...],
}
```

The generator is deterministic and no-cost. `source_max_text_chars` controls the
existing source-row adapter text cap, matching `signal_extraction` and
`social_post`. `ContentOpsExecutionServices` gets an `ad_copy` slot and dispatch
handler that passes `scope`, `target_mode`, `source_material`, `limit`, and
`max_text_chars` to the service.

## Intentional

- No persistence or generated-asset review branch. This first slice proves the
  output can run from source evidence; durable review/export handoff should
  follow only after the reviewer accepts the output shape.
- No package-default enrollment yet. Review/competitive package defaults remain
  unchanged until the `ad_copy` output is executable and tested.
- No #1268 output-variations work. `variant_count` and prompt-angle fan-out are
  a separate plan-only lane and remain untouched.
- No LLM ad-writing prompt. This slice keeps copy bounded to real review/source
  evidence and avoids paid-media hallucination risk.

## Deferred

- Add `ad_copy` to review and competitive input-package defaults in the next
  ad-copy handoff slice.
- Persist generated ad-copy drafts into the generated-assets review queue after
  the output shape is accepted.
- Stat/quote card output remains the next marketer asset family after ad copy.
- LLM/style/channel variants and #1268-style output variations remain future
  product polish.

## Parked hardening

None.

## Verification

- Passed: `pytest tests/test_extracted_ad_copy_generation.py tests/test_extracted_content_control_surfaces.py tests/test_extracted_content_generation_plan.py tests/test_extracted_content_ops_execution.py tests/test_extracted_content_reasoning_policy.py tests/test_extracted_content_control_surface_api.py tests/test_atlas_content_ops_execution_services.py -q` (316 passed, 1 skipped)
- Passed: `pytest tests/test_extracted_content_ops_execution.py::test_services_with_reasoning_context_preserves_ad_copy_service tests/test_extracted_content_ops_execution.py::test_execute_runs_ad_copy_service_from_source_material tests/test_extracted_ad_copy_generation.py -q` (6 passed)
- Passed: `cd atlas-intel-ui && npm run test:content-ops-ingestion-limits` (10 passed)
- Passed: `python -m json.tool atlas-intel-ui/src/api/__fixtures__/contentOps/catalog.json >/dev/null`
- Passed: `python -m py_compile extracted_content_pipeline/ad_copy_generation.py extracted_content_pipeline/control_surfaces.py extracted_content_pipeline/generation_plan.py extracted_content_pipeline/content_ops_execution.py extracted_content_pipeline/reasoning_policy.py atlas_brain/_content_ops_services.py tests/test_extracted_ad_copy_generation.py tests/test_extracted_content_control_surfaces.py tests/test_extracted_content_generation_plan.py tests/test_extracted_content_ops_execution.py tests/test_extracted_content_reasoning_policy.py tests/test_extracted_content_control_surface_api.py tests/test_atlas_content_ops_execution_services.py`
- Passed: `git diff --check`
- Passed: `python scripts/audit_extracted_pipeline_ci_enrollment.py --atlas-brain-tests-from origin/main` (OK: 144 matching tests are enrolled.)
- Passed: `bash scripts/validate_extracted_content_pipeline.sh`
- Passed: `python extracted/_shared/scripts/forbid_atlas_reasoning_imports.py extracted_content_pipeline`
- Passed: `python scripts/audit_extracted_standalone.py --fail-on-debt`
- Passed: `bash scripts/check_ascii_python.sh`
- Passed: `cd atlas-intel-ui && npm run lint`
- Passed: `cd atlas-intel-ui && npm run build`
- Passed: `bash scripts/run_extracted_pipeline_checks.sh` (2983 passed, 10 skipped, 1 warning)
- Passed: `bash scripts/local_pr_review.sh --current-pr-body-file /tmp/content-ops-marketer-ad-copy-output-pr-body.md`

## Estimated diff size

Actual: 17 files, +763 / -2. This is above the 400 LOC soft cap because the
first executable ad-copy output is not reviewable unless the service, catalog,
planning, dispatch, host wiring, fixture, and tests land together.

| Area | Estimated LOC |
|---|---:|
| Ad-copy service | ~239 |
| Catalog/plan/executor/host/reasoning/docs/fixture wiring | ~86 |
| Focused tests | ~291 |
| Plan doc | ~143 |
| **Total** | **~765** |

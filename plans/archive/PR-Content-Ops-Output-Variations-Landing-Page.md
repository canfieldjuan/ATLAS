# PR-Content-Ops-Output-Variations-Landing-Page

## Why this slice exists

The merged `PR-Content-Ops-Output-Variations.md` plan called for output
variations across blog posts, landing pages, and sales briefs, with an explicit
fallback to land blog-only first if the full parity slice exceeded the 400 LOC
budget. PR #1347 took that fallback and shipped blog variants through preview,
plan, execution, and prompt metadata. Its Deferred section names
landing-page `variant_angle` parity as the immediate follow-up.

This PR adds that second generator. It keeps the slice narrow: landing-page
preview/plan/execution now honors `variant_count`, and the landing-page
generator gets the same optional `variant_angle` prompt/metadata threading that
blog posts already have. Sales-brief parity and a shared multi-generator fan-out
refactor remain deferred so this PR can prove the landing-page behavior without
rewriting the already-reviewed blog path.

The full diff is over the 400 LOC soft cap because the slice includes the plan
doc plus focused R2/R12 regression tests for preview cost, plan metadata,
execution fan-out, per-item failure isolation, all-raising failure status, and
prompt metadata. The implementation code stays well under 400 LOC.

## Scope (this PR)

Ownership lane: content-ops/output-variations/landing-page
Slice phase: Vertical slice

1. Make landing-page preview cost and generation-plan metadata scale by the
   selected deterministic variant angles when `variant_count > 1`.
2. Fan out landing-page execution once per selected angle, aggregate per-angle
   results, preserve per-item failure isolation, and fail the step/run when all
   requested landing-page variants raise.
3. Add `variant_angle: str | None = None` to
   `LandingPageGenerationService.generate(...)`, thread it into the LLM prompt,
   LLM metadata, parsed metadata, and saved draft metadata.
4. Keep `variant_count == 1` as the existing single landing-page call and keep
   sales-brief behavior unchanged in this slice.

### Review Contract
- Acceptance criteria:
  - [ ] Landing-page preview cost multiplies by the normalized variant count;
        sales brief remains unscaled until its own parity slice.
  - [ ] Landing-page generation-plan steps expose `variant_count` and
        `variant_angles` when variants are requested.
  - [ ] `execute_content_ops_request` calls the landing-page service once per
        selected angle, passes each angle instruction as `variant_angle`, and
        aggregates `variant_results`, counts, saved ids, errors, and warnings.
  - [ ] One failed landing-page variant does not abort sibling variants.
  - [ ] If every landing-page variant raises an exception, the step/run fails
        while preserving the aggregate diagnostic result.
  - [ ] `LandingPageGenerationService.generate(..., variant_angle=...)`
        injects the angle into the user prompt and preserves it in LLM call
        metadata and saved draft metadata; `None` is a no-op.
- Affected surfaces: extracted package preview cost, generation-plan metadata,
  landing-page execution dispatcher, landing-page prompt/metadata, tests.
- Risk areas: cost accounting, execution result contract, prompt truthfulness,
  backward compatibility for existing landing-page callers.
- Reviewer rules triggered: R1, R2, R5, R10, R12.

### Files touched

- `extracted_content_pipeline/content_ops_execution.py`
- `extracted_content_pipeline/control_surfaces.py`
- `extracted_content_pipeline/generation_plan.py`
- `extracted_content_pipeline/landing_page_generation.py`
- `plans/PR-Content-Ops-Output-Variations-Landing-Page.md`
- `tests/test_extracted_content_control_surfaces.py`
- `tests/test_extracted_content_generation_plan.py`
- `tests/test_extracted_content_ops_execution.py`
- `tests/test_extracted_landing_page_generation_smoke.py`

## Mechanism

Preview cost expands the existing item multiplier from blog-only to
blog-or-landing-page. The generation plan reuses the deterministic
`selected_variant_angles` catalogue and adds `variant_count` /
`variant_angles` to landing-page step config when the count is greater than one.

Runtime fan-out stays landing-page-local for this slice. `_dispatch_landing_page`
preserves the existing single service call for `variant_count == 1`; for
multi-variant requests it calls the landing-page service once per selected angle
with `variant_angle=angle.instruction`, tags each per-call result with
`angle.as_dict()`, and returns the same aggregate shape the blog variant path
uses (`variant_count`, `variant_results`, total counts, saved ids, errors, and
optional warnings). Caught per-angle exceptions become tagged errors so sibling
variants still run. If every requested landing-page variant raises, the
dispatcher raises `_ContentOpsStepResultFailure("all_landing_page_variants_failed", aggregate)`
so the step/run status is failed while the aggregate remains attached to the
failed step.

The landing-page generator accepts an optional `variant_angle` string. When
present, `_landing_page_user_prompt` adds a short "Variant angle" instruction
that tells the model to change framing while preserving the same campaign facts
and supplied evidence. The same angle is sent in LLM metadata, carried on the
parsed payload, and written into the saved `LandingPageDraft.metadata`. Empty or
`None` values are ignored so existing callers keep current behavior.

## Intentional

- This is landing-page parity, not the shared fan-out refactor. The PR touches
  the minimum execution path required to prove landing-page variants without
  rewriting blog variant code that just merged.
- Sales-brief `variant_angle` parity remains separate. That generator has a
  different input shape and should get its own focused tests.
- The quality gate stays inside `LandingPageGenerationService`. The executor
  does not duplicate landing-page quality logic; it only aggregates the
  generated/skipped/errors contract the service already returns.
- Variant angle changes framing only. The prompt explicitly preserves the same
  campaign facts and supplied evidence so variants do not become new claims.

## Deferred

- Sales-brief `variant_angle` parity.
- Shared multi-generator fan-out/aggregation refactor after landing-page parity
  is reviewed.
- Persistent variant grouping (`variant_group_id`) and a past-run variants view.
- Auto-ranking / recommended winner and A/B serving/analytics.

Parked hardening: none.

## Verification

- Command: pytest tests/test_extracted_content_control_surfaces.py tests/test_extracted_content_generation_plan.py tests/test_extracted_content_ops_execution.py tests/test_extracted_landing_page_generation_smoke.py -q -- PASS, 180 tests.
- Command: bash scripts/validate_extracted_content_pipeline.sh -- PASS.
- Command: python extracted/_shared/scripts/forbid_atlas_reasoning_imports.py extracted_content_pipeline -- PASS.
- Command: python scripts/audit_extracted_standalone.py --fail-on-debt -- PASS.
- Command: bash scripts/check_ascii_python.sh -- PASS.
- Command: bash scripts/run_extracted_pipeline_checks.sh -- PASS, 3,237 passed / 10 skipped.

## Estimated diff size

| File | LOC |
|---|---:|
| `extracted_content_pipeline/content_ops_execution.py` | 98 |
| `extracted_content_pipeline/control_surfaces.py` | 2 |
| `extracted_content_pipeline/generation_plan.py` | 5 |
| `extracted_content_pipeline/landing_page_generation.py` | 46 |
| `plans/PR-Content-Ops-Output-Variations-Landing-Page.md` | 144 |
| `tests/test_extracted_content_control_surfaces.py` | 17 |
| `tests/test_extracted_content_generation_plan.py` | 20 |
| `tests/test_extracted_content_ops_execution.py` | 150 |
| `tests/test_extracted_landing_page_generation_smoke.py` | 35 |
| **Total** | **517** |

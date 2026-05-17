# Content Ops Landing Reasoning Parity

## Why this slice exists

Landing pages already support reasoning-aware generation through
`LandingPageGenerationService.with_reasoning_context`, and the reasoning
catalog allows `multi_pass_structured` for `landing_page`. The packaged runtime
allowlist still excludes `landing_page`, so `/plan` and `/execute` reject a
valid catalog choice.

This slice closes that policy/runtime mismatch without adding a new generator.

## Scope

1. Add `landing_page` to packaged structured reasoning runtime outputs.
2. Thread structured reasoning config into landing-page generation plan steps.
3. Update control-surface validation error text and tests.
4. Clean the merged #565 inflight row and claim this slice.

### Files touched

- `docs/extraction/coordination/inflight.md`
- `docs/extraction/coordination/queue.md`
- `plans/PR-ContentOps-Landing-Reasoning-Parity.md`
- `extracted_content_pipeline/reasoning_policy.py`
- `extracted_content_pipeline/generation_plan.py`
- `extracted_content_pipeline/api/control_surfaces.py`
- `extracted_content_pipeline/STATUS.md`
- `tests/test_extracted_content_reasoning_policy.py`
- `tests/test_extracted_content_generation_plan.py`
- `tests/test_extracted_content_control_surface_api.py`

## Mechanism

`PACKAGED_REASONING_RUNTIME_OUTPUTS` becomes the shared source of truth for
blog posts, reports, landing pages, and sales briefs. Landing pages accept only
`multi_pass_structured`, matching their catalog support and avoiding strict
falsification semantics that the product policy does not advertise.

The generation-plan landing-page step includes the same informational reasoning
metadata as other packaged outputs when a runtime preset is requested. The
control-surface API then builds a `MultiPassCampaignReasoningProvider` for the
landing-page service through the existing `with_reasoning_context` seam.

## Intentional

- No change to the default landing-page preset (`single_pass`).
- No strict landing-page reasoning support.
- No new LLM/provider wiring; this reuses the existing packaged structured
  provider path.

## Deferred

- Per-output landing-page narrative pack tuning. The default structured pack is
  used until a dedicated landing-page pack is designed.
- UI changes for selecting reasoning presets.

## Verification

- python -m py_compile on touched Python files - passed.
- pytest tests/test_extracted_content_reasoning_policy.py
  tests/test_extracted_content_generation_plan.py
  tests/test_extracted_content_control_surface_api.py - 105 passed.
- git diff --check - passed.
- bash scripts/local_pr_review.sh - passed after commit.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Coordination docs | ~10 |
| Plan doc | ~80 |
| Runtime policy and validation | ~20 |
| Tests | ~55 |
| Status docs | ~5 |
| **Total** | ~170 |

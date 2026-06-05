# Content Ops Email Campaign Reasoning Parity

## Why this slice exists

`email_campaign` is the core AI Content Ops output and already supports
reasoning-aware generation through `CampaignGenerationService.with_reasoning_context`.
The packaged `/content-ops/execute` runtime still excludes it from structured
reasoning construction, so hosts can provide campaign context manually but
cannot opt into the same packaged `multi_pass_structured` path that newer asset
outputs use.

This slice closes that policy/runtime mismatch without adding a new generator.

## Scope

1. Add `email_campaign` to packaged structured reasoning runtime outputs.
2. Thread structured reasoning config into email campaign generation plan steps.
3. Keep strict/falsification support out of scope for campaigns.
4. Update control-surface validation error text and tests.
5. Clean the merged #566 inflight row and claim this slice.

### Files touched

- `docs/extraction/coordination/inflight.md`
- `docs/extraction/coordination/queue.md`
- `plans/PR-ContentOps-Email-Packaged-Reasoning.md`
- `extracted_content_pipeline/reasoning_policy.py`
- `extracted_content_pipeline/generation_plan.py`
- `extracted_content_pipeline/api/control_surfaces.py`
- `extracted_content_pipeline/STATUS.md`
- `tests/test_extracted_content_reasoning_policy.py`
- `tests/test_extracted_content_generation_plan.py`
- `tests/test_extracted_content_control_surface_api.py`

## Mechanism

`PACKAGED_REASONING_RUNTIME_OUTPUTS` becomes the shared source of truth for
email campaigns, blog posts, reports, landing pages, and sales briefs. Email
campaigns accept only `multi_pass_structured`, matching the campaign service's
existing reasoning-context seam while avoiding strict falsification semantics
that the product policy does not advertise for campaigns.

The generation-plan email campaign step includes the same informational
reasoning metadata as other packaged outputs when a runtime preset is requested.
The control-surface API then builds a `MultiPassCampaignReasoningProvider` for
the campaign service through the existing `with_reasoning_context` seam.

## Intentional

- No change to the default email campaign preset (`single_pass`).
- No strict email campaign reasoning support.
- No new LLM/provider wiring; this reuses the existing packaged structured
  provider path.
- No service signature changes.

## Deferred

- Campaign-specific narrative pack tuning. The default structured pack is used
  until a dedicated campaign pack is designed.
- UI changes for selecting reasoning presets.

## Verification

- Python compile on touched Python files - passed.
- Focused policy, plan, and API tests - 108 passed.
- Git diff whitespace check - passed.
- Local PR review - pending after commit.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Coordination docs | ~10 |
| Plan doc | ~80 |
| Runtime policy and validation | ~25 |
| Tests | ~70 |
| Status docs | ~5 |
| **Total** | ~190 |

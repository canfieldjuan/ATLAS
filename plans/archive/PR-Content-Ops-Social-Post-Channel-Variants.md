# PR-Content-Ops-Social-Post-Channel-Variants

## Why this slice exists

PR-Content-Ops-Social-Post-LLM-Voice made `social_post` consume stored brand
voice, but deliberately kept the output to one LinkedIn draft per source row.
Its Deferred section named `PR-Content-Ops-Social-Post-Channel-Variants` as
the next product step: social posts need platform-specific variants, not one
generic card with a hardcoded `channel="linkedin"`.

This slice keeps the default behavior unchanged while adding the thinnest
request-level channel control: when the request supplies social channels, the
social-post service emits one source-backed draft per selected platform and the
preview budget scales the brand-voice LLM estimate by the selected channel
count.

The estimated diff is over the 400 LOC soft cap because the slice crosses the
minimum contract surfaces together: generator behavior, plan metadata,
execution dispatch, cost preview, and focused tests for each boundary. Splitting
these would leave a channel selector path without either cost gating or a real
execution proof.

## Scope (this PR)

Ownership lane: content-ops/brand-voice/social-post-channel-variants
Slice phase: Vertical slice

1. Add normalized `social_channels` / `social_post_channels` input handling for
   `social_post`; the default remains `("linkedin",)`.
2. Expand deterministic social-post generation to one draft per selected
   channel for each usable source row, with stable channel-specific ids and
   copy constraints.
3. Keep the brand-voice LLM rewrite path fail-closed, but call it once per
   selected channel and preserve the requested channel on the rewritten draft.
4. Thread selected channels through generation-plan metadata and the execution
   dispatcher into `SocialPostGenerationService.generate(...)`.
5. Scale the brand-voice social-post preview cost by selected channel count so
   `max_cost_usd` still protects multi-channel LLM rewrites.
6. Add focused tests for default compatibility, channel expansion, invalid
   channel rejection, dispatcher threading, and cost scaling.

### Review Contract

- Acceptance criteria: default `social_post` still emits the existing LinkedIn
  draft; requested social channels emit one draft per channel; invalid channel
  ids fail closed; brand-voice rewrites preserve requested channels; preview
  cost scales by channel count only for brand-voice social posts.
- Affected surfaces: extracted social-post generator, generation-plan config,
  execution dispatcher, control-surface preview cost, package tests.
- Risk areas: backcompat, cost gating, LLM prompt/metadata shape, invalid input
  validation.
- Reviewer rules triggered: R1 (requirements match the plan/test contract),
  R10 (channel normalization and dispatch stay centralized and maintainable).

### Files touched

- `extracted_content_pipeline/content_ops_execution.py`
- `extracted_content_pipeline/control_surfaces.py`
- `extracted_content_pipeline/generation_plan.py`
- `extracted_content_pipeline/social_post_generation.py`
- `plans/PR-Content-Ops-Social-Post-Channel-Variants.md`
- `tests/test_extracted_content_control_surfaces.py`
- `tests/test_extracted_content_generation_plan.py`
- `tests/test_extracted_content_ops_execution.py`
- `tests/test_extracted_social_post_generation.py`

## Mechanism

`SocialPostGenerationConfig` gains a normalized `channels` tuple. The service's
`generate(...)` accepts an optional `channels` override, normalizes it through a
single helper, and passes the result into `_generate_social_posts(...)`. The
default is still `("linkedin",)`, so existing callers and tests keep the same
single-post behavior and current text.

For multi-channel requests, `_generate_social_posts(...)` treats `limit` as the
source-row limit and emits a draft per selected channel. Non-default multi-
channel ids include the channel suffix to avoid duplicate UI ids for the same
source row. Each channel receives a bounded deterministic text variant; the
`x` variant is capped tighter than the configured post bound.

Brand-voice rewriting already operates over the deterministic posts. This slice
keeps that shape: after deterministic expansion, the LLM rewrite path is called
once per generated channel draft. The rewrite prompt and metadata include the
requested channel, and the rewritten result preserves the source draft's
normalized channel rather than letting a model silently drift a LinkedIn draft
into another platform.

The generation plan records `channels` in the `social_post` step config, the
executor passes that config into the social-post service, and the control-
surface estimator multiplies the brand-voice social-post unit estimate by the
same normalized channel count. Deterministic social posts remain zero-estimate.

## Intentional

- This does not add a new output id. `social_post` remains the product output;
  `inputs.social_channels` only selects platform variants.
- `inputs.channels` remains campaign-owned. This slice uses
  `social_channels` / `social_post_channels` so email campaign channel
  selection and social platform selection do not collide in mixed-output runs.
- No frontend selector in this slice; this is the API/package path that the UI
  can wire in a follow-up.

## Deferred

- `PR-Content-Ops-Social-Post-Channel-Selector-UI`: expose `social_channels`
  in the New Run UI.
- `PR-Content-Ops-Brand-Voice-Settings-Page`: optional standalone management
  page if the inline New Run panel becomes too dense.

Parked hardening: none.

## Verification

- python -m py_compile extracted_content_pipeline/social_post_generation.py extracted_content_pipeline/generation_plan.py extracted_content_pipeline/content_ops_execution.py extracted_content_pipeline/control_surfaces.py tests/test_extracted_social_post_generation.py tests/test_extracted_content_generation_plan.py tests/test_extracted_content_ops_execution.py tests/test_extracted_content_control_surfaces.py
  - Pass.
- pytest tests/test_extracted_social_post_generation.py tests/test_extracted_content_generation_plan.py tests/test_extracted_content_ops_execution.py tests/test_extracted_content_control_surfaces.py -q
  - `180 passed in 0.28s`.
- bash scripts/validate_extracted_content_pipeline.sh
  - Pass.
- python extracted/_shared/scripts/forbid_atlas_reasoning_imports.py extracted_content_pipeline
  - Pass.
- python scripts/audit_extracted_standalone.py --fail-on-debt
  - Pass.
- bash scripts/check_ascii_python.sh
  - Pass.
- bash extracted/_shared/scripts/sync_extracted.sh extracted_content_pipeline
  - Pass; refreshed 46 synced files with no additional working-tree changes.
- bash scripts/run_extracted_pipeline_checks.sh
  - `3220 passed, 10 skipped, 1 warning in 55.92s`.

## Estimated diff size

| File | LOC |
|---|---:|
| `extracted_content_pipeline/content_ops_execution.py` | 1 |
| `extracted_content_pipeline/control_surfaces.py` | 15 |
| `extracted_content_pipeline/generation_plan.py` | 15 |
| `extracted_content_pipeline/social_post_generation.py` | 186 |
| `plans/PR-Content-Ops-Social-Post-Channel-Variants.md` | 144 |
| `tests/test_extracted_content_control_surfaces.py` | 34 |
| `tests/test_extracted_content_generation_plan.py` | 24 |
| `tests/test_extracted_content_ops_execution.py` | 31 |
| `tests/test_extracted_social_post_generation.py` | 105 |
| **Total** | **555** |

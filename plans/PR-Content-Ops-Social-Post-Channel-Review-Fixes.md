# PR-Content-Ops-Social-Post-Channel-Review-Fixes

## Why this slice exists

PR #1340 merged the social-post channel-variant path, then the review posted
two actionable MAJOR findings:

1. Unsupported `social_channels` values now flow through the preview cost path,
   where `normalize_social_post_channels(...)` can raise `ValueError`. The API
   `/plan` and `/execute` routes already turn request-shape `ValueError`s into
   400s, but `/preview` is intended to be a non-throwing preflight surface and
   currently lets this new invalid input become an unhandled preview failure.
2. The core channel behavior was correct, but several load-bearing behaviors
   were not mutation-pinned: source-row limit semantics, partial brand-voice
   rewrite failure isolation, and channel-specific caps/coverage.

This slice closes only those review findings before moving to the next product
feature.

## Scope (this PR)

Ownership lane: content-ops/brand-voice/social-post-channel-variants
Slice phase: Robust testing

1. Keep `/preview` non-throwing for unsupported social-post channels by turning
   the normalization error into a warning and `can_run=False`.
2. Add focused tests proving invalid preview channels do not raise or allow a
   run.
3. Add mutation-pinning tests for multi-channel limit semantics, partial LLM
   rewrite failure isolation, and channel-specific caps/coverage.
4. Remove the unreachable defensive fallback in `_post_body_for_channel(...)`
   that the review marked as a NIT.

### Review Contract

- Acceptance criteria: unsupported preview channels return a blocked preview
  with a warning; generator `limit` caps source rows under multi-channel fanout;
  one failed brand-voice channel rewrite does not abort sibling channels; X
  brand-voice output is capped at 280 chars; Threads deterministic generation
  has direct coverage; the unreachable channel body fallback is gone.
- Affected surfaces: extracted control-surface preview, extracted social-post
  generator tests, plan doc.
- Risk areas: preview API compatibility, invalid input handling, regression
  test strength.
- Reviewer rules triggered: R1 (review findings map to tests and behavior),
  R10 (validator/normalizer failure branch is proven).

### Files touched

- `extracted_content_pipeline/control_surfaces.py`
- `extracted_content_pipeline/social_post_generation.py`
- `plans/PR-Content-Ops-Social-Post-Channel-Review-Fixes.md`
- `tests/test_extracted_content_control_surface_api.py`
- `tests/test_extracted_content_control_surfaces.py`
- `tests/test_extracted_social_post_generation.py`

## Mechanism

`preview_control_surface(...)` already owns the warning/blocking decision for
unknown presets, outputs, ingestion profiles, missing inputs, and cost-budget
warnings. This slice keeps unsupported social-post channels in that same
preflight model: validate the social-post channel multiplier before estimating
cost, catch the channel normalization `ValueError`, append a warning, block the
`social_post` output, and keep `can_run=False`. The execution path still fails
closed through the existing `/plan` and `/execute` `ValueError -> 400`
handling.

Tests then pin the behaviors that #1340 introduced:

- multi-channel generation with `limit=2` and three usable rows proves limit is
  a source-row cap, not a total-post cap;
- a first-channel LLM exception followed by a valid second-channel response
  proves per-channel failure isolation and no deterministic fallback;
- an overlong X rewrite proves the 280-char cap is active;
- a Threads deterministic request proves that channel's body path is exercised.

The unreachable `_post_body_for_channel(...)` fallback becomes an explicit
invariant failure with a comment naming the upstream normalization guarantee.

## Intentional

- This does not reopen the product contract from #1340 or add frontend selector
  wiring.
- Preview warnings keep invalid social-channel input in the same UX category as
  other preflight-only blockers instead of throwing an API error from preview.
- `/plan` and `/execute` are left unchanged; they already return 400 for the
  same invalid request shape.

## Deferred

- `PR-Content-Ops-Social-Post-Channel-Selector-UI`: expose `social_channels`
  in the New Run UI after these review findings are fixed.
- `PR-Content-Ops-Brand-Voice-Settings-Page`: optional standalone management
  page if the inline New Run panel becomes too dense.

Parked hardening: none.

## Verification

- python -m py_compile extracted_content_pipeline/control_surfaces.py extracted_content_pipeline/social_post_generation.py tests/test_extracted_content_control_surfaces.py tests/test_extracted_content_control_surface_api.py tests/test_extracted_social_post_generation.py
  - Pass.
- pytest tests/test_extracted_social_post_generation.py tests/test_extracted_content_control_surfaces.py tests/test_extracted_content_control_surface_api.py -q
  - `199 passed, 1 skipped in 3.77s`.
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
  - `3222 passed, 10 skipped, 1 warning in 56.07s`.

## Estimated diff size

| File | LOC |
|---|---:|
| `extracted_content_pipeline/control_surfaces.py` | 7 |
| `extracted_content_pipeline/social_post_generation.py` | 3 |
| `plans/PR-Content-Ops-Social-Post-Channel-Review-Fixes.md` | 127 |
| `tests/test_extracted_content_control_surface_api.py` | 27 |
| `tests/test_extracted_content_control_surfaces.py` | 22 |
| `tests/test_extracted_social_post_generation.py` | 142 |
| **Total** | **328** |

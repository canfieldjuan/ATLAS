# PR: Content Ops Reasoning Preset Catalog

## Why this slice exists

PR #553 documented the Content Ops reasoning-policy boundary. This slice adds
the pure preset catalog for selecting reasoning depth per output without
constructing providers or changing generation behavior.

## Scope

Add `reasoning_policy` with the six audit preset IDs, per-output defaults,
supported-preset validation helpers, and docs/status updates.

### Files Touched

`extracted_content_pipeline/reasoning_policy.py`,
`tests/test_extracted_content_reasoning_policy.py`,
`docs/audits/ai_content_ops_deferred_backlog_2026-05-11.md`,
`extracted_content_pipeline/STATUS.md`,
`extracted_content_pipeline/manifest.json`,
`plans/PR-Content-Ops-Reasoning-Preset-Catalog.md`

## Mechanism

`ReasoningPresetDefinition` describes each preset, `OutputReasoningPolicy` maps
output IDs to supported presets/defaults, and the helpers resolve policies for
future runtime wiring. No provider construction happens here.

## Intentional

No runtime behavior, FastAPI schema, provider-construction, cache, state,
falsification, narrative-plan, or validation wiring changes.

## Deferred

- API/control-surface exposure.
- Report/sales-brief structured-reasoning construction.
- Validation metadata surfacing and blog-specific narrative packs.

## Verification

- Run focused reasoning-policy tests.
- Run the local review script at `scripts/local_pr_review.sh`.

## Estimated Diff Size

| Area | Estimated LOC |
|---|---:|
| Policy module | ~200 |
| Tests | ~130 |
| Docs/status/plan | ~70 |
| **Total** | ~400 |

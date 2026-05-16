# Content Ops Reasoning Runtime Invariants

## Why this slice exists

The reasoning preset catalog can describe broader future policy than the
packaged runtime currently executes. Before the next reasoning-depth slice, the
plan and execute paths need one shared runtime contract so unsupported
`reasoning_preset` requests fail consistently.

## Scope (this PR)

1. Centralize packaged reasoning runtime outputs and presets in the reasoning
   policy module.
2. Use those shared constants from generation planning and the execute API.
3. Make `/plan` reject generated reasoning presets when no packaged runtime
   output is selected, matching `/execute`.
4. Preserve mixed-output behavior: packaged reasoning applies to report/sales
   steps while non-runtime outputs continue without reasoning config.
5. Refresh status/backlog/coordination docs for the invariant slice.

### Files touched

- `docs/audits/ai_content_ops_deferred_backlog_2026-05-11.md`
- `docs/extraction/coordination/inflight.md`
- `extracted_content_pipeline/STATUS.md`
- `extracted_content_pipeline/api/control_surfaces.py`
- `extracted_content_pipeline/generation_plan.py`
- `extracted_content_pipeline/reasoning_policy.py`
- `plans/PR-Content-Ops-Reasoning-Runtime-Invariants.md`
- `tests/test_extracted_content_generation_plan.py`
- `tests/test_extracted_content_reasoning_policy.py`

## Mechanism

`PACKAGED_REASONING_RUNTIME_OUTPUTS` and
`PACKAGED_REASONING_RUNTIME_PRESETS` now live in `reasoning_policy.py`.
Generation planning validates explicit generated-reasoning requests against
those constants before building steps, while the execute API uses the same
constants when constructing packaged structured/strict providers.

## Intentional

This does not add packaged reasoning to blog posts, landing pages, or email.
Those outputs can still consume host-provided reasoning context; this slice
only prevents plan/execute drift for generated packaged reasoning presets.

## Deferred

Host-owned falsification policy wiring for strict presets remains separate
product-policy work.

## Verification

pytest tests/test_extracted_content_reasoning_policy.py
tests/test_extracted_content_generation_plan.py
tests/test_extracted_content_control_surface_api.py -> 91 passed.
py_compile -> passed. git diff check -> passed. ASCII grep on touched Python
files -> passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| **Total** | ~230 |

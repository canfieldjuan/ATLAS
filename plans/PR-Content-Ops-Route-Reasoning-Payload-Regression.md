# Content Ops Route Reasoning Payload Regression

## Why this slice exists

Recent slices added route-level reasoning provider wiring, executor-level
`reasoning.consumed_contexts`, and UI rendering for consumed contexts. Coverage
still had one boundary gap: no test proved that `POST /content-ops/execute`
returns consumed context payloads after resolving a route-level provider and
rebinding the host service bundle.

## Scope (this PR)

1. Add a route-boundary regression test for provider resolution ->
   `with_reasoning_context()` -> execution response `reasoning.consumed_contexts`.
2. Remove the merged #473 coordination row and claim this slice.

### Files touched

- `tests/test_extracted_content_control_surface_api.py`
- `docs/extraction/coordination/inflight.md`
- `plans/PR-Content-Ops-Route-Reasoning-Payload-Regression.md`

## Mechanism

The test uses the existing control-surface router directly. A fake campaign
service returns `reasoning_contexts_used=1` and a
`consumed_reasoning_contexts` row only when the route-level provider has been
attached through `with_reasoning_context()`. The assertion checks the endpoint
payload, not the lower-level executor helper.

## Intentional

- No production code change. This is a regression lock for behavior already
  shipped across prior slices.
- The fake service does not call a real provider. Provider lookup is already
  covered by the file/DB provider tests and the live-DSN check CLI.

## Deferred

- No frontend screenshot or browser test. PR #473 already covered the UI build.
- No live database setup. PR #471 owns live provider lookup.

## Verification

- `pytest tests/test_extracted_content_control_surface_api.py` -> 26 passed
- `python -m py_compile tests/test_extracted_content_control_surface_api.py` -> passed
- `git diff --check` -> passed
- ASCII byte check on edited Python file -> passed

## Estimated diff size

3 files, about +90 / -2 including this plan.

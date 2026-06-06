# Repair Atlas graph routing tests for port adapter seam

## Why this slice exists

PR #570 moved the graph reason node onto the `AtlasLLMClient` port-adapter
path. A focused check of the older Atlas graph routing tests found four stale
tests still monkeypatching `atlas_brain.reasoning.graph._llm_generate`. Those
tests now exercise the fallback path instead of the intended LLM response path.

## Scope

Repair the stale tests so they provide fake LLM services with a `chat(...)`
method, matching the current `AtlasLLMClient` runtime contract.

### Files touched

- `docs/extraction/coordination/inflight.md`
- `docs/extraction/coordination/queue.md`
- `docs/extraction/coordination/state.md`
- `plans/PR-Reasoning-Core-Graph-Routing-Test-Seam-2026-05-17.md`
- `tests/test_reasoning_graph_routing.py`
- `tests/test_reasoning_graph_summary.py`

## Mechanism

- Add small fake chat-service helpers in the affected tests.
- Replace stale `_llm_generate` monkeypatches with `get_pipeline_llm` /
  `_resolve_graph_llm` fakes returning those chat-service helpers.
- Keep the existing assertions: workload routing, model override behavior,
  parsed graph outputs, and summary sanitization.
- Close the merged #570 coordination row and claim this repair slice.

## Intentional

- No production behavior changes.
- No changes to `AtlasLLMClient`.
- No changes to the reasoning graph node APIs.

## Deferred

- Further graph/state wrapper split work remains after the tests match the
  current port seam.

## Verification

- Focused Python compile for the two edited test files.
- Focused pytest for graph routing and graph summary tests.
- Focused regression sweep including graph node and graph wiring tests.
- Git whitespace check.
- Local PR review script.

## Estimated diff size

| Area | LOC |
|---|---:|
| Coordination + plan | ~65 |
| Test seam repair | ~75 |
| **Total** | ~140 |

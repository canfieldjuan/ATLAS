# PR: Reasoning Core Reason Node

## Why this slice exists

`extracted_reasoning_core.graph_nodes` already owns triage and synthesis node
execution, but Atlas still owns the reason-node LLM call, JSON parsing, token
accounting, and fallback behavior. That leaves one graph execution contract
embedded in Atlas even though the provider port and helper machinery are
already in core.

## Scope

Promote the reason-node execution contract into core while keeping Atlas's
host-specific prompt construction in Atlas.

### Files touched

- `atlas_brain/reasoning/graph.py`
- `docs/extraction/coordination/inflight.md`
- `docs/extraction/coordination/queue.md`
- `docs/extraction/coordination/state.md`
- `extracted_reasoning_core/graph_nodes.py`
- `plans/PR-Reasoning-Core-Node-Reason-2026-05-17.md`
- `tests/test_extracted_reasoning_core_graph_nodes.py`

## Mechanism

- Add `node_reason(...)` to `extracted_reasoning_core.graph_nodes`.
- Have `node_reason(...)` call `complete_with_json`, accumulate token usage,
  write `reasoning_output`, project parsed `connections/actions/rationale`, and
  preserve the unparseable-output fallback that forces notification.
- Update Atlas `_node_reason` to build the same prompt and delegate execution
  to core through `AtlasLLMClient`.
- Add focused tests for happy path, missing LLM, LLM exception, and unparseable
  output.

## Intentional

- Atlas still builds the prompt from Atlas-only state fields like
  `crm_context`, `email_history`, `voice_turns`, and `b2b_churn`.
- Core does not import Atlas prompts, settings, or LLM routing.
- Existing graph orchestration and action execution stay unchanged.

## Deferred

- No graph-state TypedDict slimming.
- No migration of Atlas context aggregation, lock checks, action execution, or
  notification delivery.
- No change to available reasoning actions or prompt text.

## Verification

- Focused graph-node tests pass.
- Atlas graph wiring tests pass.
- Reasoning-core manifest and smoke checks pass.
- Local PR review passes.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Core node implementation | ~65 |
| Atlas wrapper delegation | ~45 |
| Tests | ~95 |
| Coordination docs | ~10 |
| Plan doc | ~75 |
| **Total** | ~290 |

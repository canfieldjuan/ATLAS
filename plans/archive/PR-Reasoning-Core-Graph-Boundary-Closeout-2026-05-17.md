# Reasoning Core Graph Boundary Closeout

## Why this slice exists

PRs #570, #571, and #572 moved the reusable graph-node LLM contracts into
`extracted_reasoning_core` and routed Atlas reflection through the same port
adapter seam. The old boundary audit still says to continue the graph/state
wrapper split, which now risks extracting Atlas host orchestration by inertia.

This slice closes the planning loop: document what core owns, what Atlas still
owns intentionally, and the concrete stop rule for reopening graph extraction.

## Scope (this PR)

1. Mark PR-C8 as merged and claim PR-C9 in the coordination queue.
2. Replace the stale in-flight row with this documentation slice.
3. Update per-product state so the next milestone is graph boundary closeout,
   not more blind graph extraction.
4. Amend the current-state audit with the post-#570/#571/#572 graph boundary.
5. Record the decision in the coordination decisions log.

### Files touched

- `docs/extraction/coordination/inflight.md`
- `docs/extraction/coordination/queue.md`
- `docs/extraction/coordination/state.md`
- `docs/extraction/coordination/decisions.md`
- `docs/extraction/reasoning_core_current_state_audit_2026-05-17.md`
- `plans/PR-Reasoning-Core-Graph-Boundary-Closeout-2026-05-17.md`

## Mechanism

This is a documentation-only slice. It reconciles the current code state after
the graph-node and reflection-port work:

- Core owns `graph_helpers`, `graph_nodes`, package graph/state substrate, and
  reusable LLM-call contracts.
- Atlas owns host LLM resolution, prompt assembly for Atlas events, LangGraph
  routing, context aggregation, locks, actions, notifications, and reflection
  orchestration.

The audit now includes a stop rule. Future graph work should start only from a
specific product need, such as a non-Atlas LLM/workload adapter contract or a
shared state model needed by a product runtime.

## Intentional

- No production code changes. The prior code slices already made the runtime
  boundary change.
- No tests added. Documentation-only coordination changes are verified through
  local review and diff checks.
- Reflection remains Atlas-side. It now uses the shared LLM port adapter, but
  its analysis target is the Atlas graph state, not a product-neutral core
  primitive.

## Deferred

- If a product needs its own event graph, add a focused adapter-contract slice
  instead of moving Atlas graph orchestration wholesale.
- If core state shape becomes a blocker for a product runtime, split Atlas-only
  state fields from shared state in a dedicated state-slimming PR.
- Continue AI Content Ops reasoning execution work from product needs, not from
  the old graph/state checklist.

## Verification

- `git diff --check`
- Local PR review bundle via `scripts/local_pr_review.sh`

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Coordination docs | ~15 |
| Current-state audit amendment | ~45 |
| Plan doc | ~65 |
| **Total** | ~125 |

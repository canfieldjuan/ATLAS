# Reasoning Core Current-State Reset

## Why this slice exists

The coordination queue and the 2026-05-03 reasoning boundary audit still point
at the old PR 4-7 backlog as if semantic cache split, pack registry, graph/state
ports, and product migration guards were untouched. Current main already has
large parts of those tracks. Without a reset doc, the next reasoning-core slice
risks repeating shipped work instead of targeting the remaining gap.

## Scope (this PR)

1. Update coordination state so other sessions know this branch owns the
   reasoning-core current-state reset.
2. Add a current-state audit that lists the shipped reasoning-core surfaces and
   the remaining gaps.
3. Amend the older boundary audit with a pointer to the current-state audit.
4. Recommend the next code slice without changing runtime code.

### Files touched

- `docs/extraction/coordination/inflight.md`
- `docs/extraction/coordination/state.md`
- `docs/extraction/coordination/queue.md`
- `docs/extraction/reasoning_core_current_state_audit_2026-05-17.md`
- `docs/extraction/reasoning_boundary_audit_2026-05-03.md`
- `plans/PR-Reasoning-Core-Current-State.md`

## Mechanism

This is a documentation-only reconciliation. It compares the old audit backlog
against the current repository surface:

- public APIs, types, ports, and semantic cache key helpers in
  `extracted_reasoning_core`;
- pack registry and graph/state modules in `extracted_reasoning_core`;
- content-pipeline and atlas wrapper tests that already route through core;
- extracted-product import-boundary guard scripts.

The resulting doc names the remaining high-value gap as the atlas-side
per-review enrichment pack split.

## Intentional

- No runtime code changes. This PR only resets planning state.
- No attempt to finish the per-review enrichment split here; that belongs in a
  focused code PR after this doc lands.
- No semantic-cache storage implementation inside core. Storage stays behind
  the existing `SemanticCacheStore` port.

## Deferred

- Per-review enrichment pack split: carve the atlas-side enrichment methods out
  of `atlas_brain.reasoning.evidence_engine` into an explicit product pack or
  document why the atlas wrapper remains the product pack.
- Graph/state slimming: decide which atlas-specific state fields belong in core
  versus product extension state.
- Remaining product migration hardening: keep tightening import boundaries only
  after the current-state map is accepted.

## Verification

- git diff --check - passed before commit.
- bash scripts/local_pr_review.sh - passed after commit.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Coordination docs | ~20 |
| Current-state audit | ~95 |
| Boundary audit pointer | ~10 |
| Plan doc | ~60 |
| **Total** | ~185 |

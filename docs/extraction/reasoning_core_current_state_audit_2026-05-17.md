# Extracted Reasoning Core Current State

Date: 2026-05-17

## Purpose

The original 2026-05-03 reasoning boundary audit is stale. It says PR 4
through PR 7 remain untouched, but current main already contains large parts of
those tracks. This note resets the next-slice decision against the code that
actually exists.

## Current Shipped Surface

`extracted_reasoning_core` now contains:

- Public API: `api.py` with `score_archetypes`, `evaluate_evidence`,
  `build_temporal_evidence`, `run_reasoning`, `continue_reasoning`,
  `check_falsification`, `build_narrative_plan`,
  `validate_reasoning_output`, `compute_evidence_hash`,
  `build_semantic_cache_key`, and `load_reasoning_pack`.
- Public types and ports: `types.py` and `ports.py`, including
  `ReasoningPorts`, `SemanticCacheStore`, `ReasoningStateStore`,
  `EventSink`, `TraceSink`, `WitnessContextPort`, `LLMClient`, and `Clock`.
- Deterministic core modules: `archetypes.py`, `temporal.py`,
  `evidence_engine.py`, `semantic_cache_keys.py`, `wedge_registry.py`,
  `tiers.py`, and `domains.py`.
- Pack/runtime modules: `pack_registry.py`, `skills/registry.py`,
  `_synthesis.py`.
- Graph/state modules: `graph.py`, `graph_nodes.py`, `graph_helpers.py`,
  and `state.py`.
- Core test coverage under `tests/test_extracted_reasoning_core_*.py`.

## Status Against The Old PR 4-7 Backlog

| Old backlog item | Current status | Notes |
|---|---|---|
| PR 4: Semantic cache split | Mostly shipped at the core boundary | `semantic_cache_keys.py`, `SemanticCacheStore`, `compute_evidence_hash`, and `build_semantic_cache_key` exist. Concrete storage remains outside core, which is the intended port split. |
| PR 5: Reasoning pack registry | Mostly shipped | `pack_registry.py` exists, and atlas single-pass prompt packs register into it. The remaining gap is per-review enrichment, still atlas-side in `atlas_brain.reasoning.evidence_engine`. |
| PR 6: Graph/state engine with ports | Partially shipped | Core graph, graph node, graph helper, and state modules exist. Atlas graph still has host-specific LLM resolution and orchestration wrappers. |
| PR 7: Product migration pass | Partially shipped | `extracted_content_pipeline.reasoning.*` wrappers point at core. Atlas wrappers for archetypes, temporal, evidence engine, and graph helpers are covered by alias tests. Import-boundary guard exists and is wired for extracted products. |

## Remaining Gaps

1. **Per-review enrichment pack split.**
   `atlas_brain.reasoning.evidence_engine.EvidenceEngine` subclasses core and
   keeps atlas-only enrichment methods: `compute_urgency`, `override_pain`,
   `derive_recommend`, `derive_price_complaint`, `derive_budget_authority`,
   and `_check_derivation_rule`. This is the clearest remaining PR 5 gap.

2. **Atlas graph/state host wrapper split.**
   `atlas_brain.reasoning.graph` still owns host-specific LLM resolution,
   timeout handling, and orchestration glue. Core owns graph helpers/nodes, but
   a full migration would be larger and riskier than the enrichment pack split.

3. **Queue/status drift.**
   Coordination docs still pointed at PR-C1 even though the code had advanced
   through semantic cache keys, pack registry, graph/state, and migration
   guards. This PR fixes that planning drift before more code work starts.

4. **Standalone-toggle status.**
   The reasoning core is moving toward the phase-2 boundary, but this audit did
   not find a dedicated `EXTRACTED_REASONING_CORE_STANDALONE` toggle. The state
   table therefore marks it as `1 -> 2 in progress`, not fully phase 2.

## Recommendation

Take the next code slice as the **per-review enrichment pack split**:

- Create a product-owned enrichment pack module outside core.
- Keep `extracted_reasoning_core.evidence_engine` slim.
- Preserve the atlas wrapper behavior so existing callers still get the same
  `EvidenceEngine` object shape.
- Add tests that prove the atlas wrapper delegates slim evidence decisions to
  core while enrichment methods live in the product pack.

Do not rebuild semantic cache, pack registry, graph helpers, or public API
surfaces from the old audit wording; those already exist.

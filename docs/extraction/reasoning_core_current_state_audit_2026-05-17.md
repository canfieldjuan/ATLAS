# Extracted Reasoning Core Current State

Date: 2026-05-17
Current-state refresh: 2026-05-17 after the #564 implementation, surfaced
during post-#575 cleanup

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
- Atlas-side product pack split:
  `atlas_brain.reasoning.review_enrichment.ReviewEnrichmentMixin` owns
  per-review enrichment methods while `atlas_brain.reasoning.evidence_engine`
  stays a wrapper over core's slim evidence engine.
- Core test coverage under `tests/test_extracted_reasoning_core_*.py`.

## Status Against The Old PR 4-7 Backlog

| Old backlog item | Current status | Notes |
|---|---|---|
| PR 4: Semantic cache split | Mostly shipped at the core boundary | `semantic_cache_keys.py`, `SemanticCacheStore`, `compute_evidence_hash`, and `build_semantic_cache_key` exist. Concrete storage remains outside core, which is the intended port split. |
| PR 5: Reasoning pack registry | Shipped for current scope | `pack_registry.py` exists, atlas single-pass prompt packs register into it, and atlas per-review enrichment now lives in `atlas_brain.reasoning.review_enrichment` with `evidence_engine` as a wrapper/mixin host. |
| PR 6: Graph/state engine with ports | Shipped to the intended boundary | Core graph, graph node, graph helper, and state modules exist. After #570-#572, Atlas graph is a host wrapper around core node contracts plus Atlas-owned orchestration. |
| PR 7: Product migration pass | Partially shipped | `extracted_content_pipeline.reasoning.*` wrappers point at core. Atlas wrappers for archetypes, temporal, evidence engine, and graph helpers are covered by alias tests. Import-boundary guard exists and is wired for extracted products. |

## Graph Boundary Closeout After PR-C6 Through PR-C8

PRs #570, #571, and #572 narrowed the graph/state gap enough that further
graph extraction should stop unless a concrete product asks for it.

Core now owns:

- `extracted_reasoning_core.graph_helpers`: prompt rendering, JSON parsing,
  token accounting, and `complete_with_json`.
- `extracted_reasoning_core.graph_nodes`: `node_triage`, `node_reason`, and
  `node_synthesize` LLM-call, parse, fallback, and usage-accumulation
  contracts.
- `extracted_reasoning_core.graph` and `state`: the package-local graph and
  state substrate used by extracted products.

Atlas now intentionally owns:

- workload-specific LLM resolution in `_resolve_graph_llm`;
- Atlas settings, timeout constants, and model override selection;
- prompt assembly for Atlas event fields in `_node_reason`;
- LangGraph orchestration and conditional routing;
- context aggregation from Atlas stores;
- entity lock checks;
- action execution and notification side effects;
- reflection as a host-side analysis pass routed through `AtlasLLMClient` and
  `complete_with_json`.

That boundary is deliberate. The remaining Atlas code depends on Atlas config,
event payloads, stores, and side effects. Moving it into core would not make AI
Content Ops more operational; it would make core own host integration behavior.

### Stop rule

Do not continue graph extraction just to match the old 2026-05-03 backlog.
Reopen this track only when a product needs one of these concrete capabilities:

1. a product-local event graph that cannot call the current core graph APIs;
2. a reusable host adapter contract for non-Atlas LLM/workload resolution;
3. a shared action/notification abstraction used by more than one product;
4. a slimmed core state model needed by a product runtime.

Until then, the higher-value reasoning work is on product-specific provider
contracts and generated reasoning contexts, not on moving Atlas wrappers.

## Gap Status

1. **Per-review enrichment pack split.**
   Closed for the current Atlas wrapper boundary. The enrichment methods
   `compute_urgency`, `override_pain`, `derive_recommend`,
   `derive_price_complaint`, `derive_budget_authority`, and
   `_check_derivation_rule` live on
   `atlas_brain.reasoning.review_enrichment.ReviewEnrichmentMixin`.
   `atlas_brain.reasoning.evidence_engine.EvidenceEngine` subclasses core and
   mixes in that atlas product pack so existing callers keep the same object
   shape while core stays slim.

2. **Atlas graph/state host wrapper split.**
   This is now intentionally closed at the host-wrapper boundary described
   above. Core owns reusable helpers and node contracts. Atlas owns workload
   resolution and side-effect orchestration.

3. **Queue/status drift.**
   Coordination docs have now advanced through semantic cache keys, pack
   registry, graph/state, migration guards, node reasoning, routing, reflection,
   and graph-boundary closeout. Future drift should be fixed with targeted
   closeout docs rather than reopening old PR-C1/PR-C5 work.

4. **Standalone-toggle status.**
   The reasoning core is moving toward the phase-2 boundary, but this audit did
   not find a dedicated `EXTRACTED_REASONING_CORE_STANDALONE` toggle. The state
   table therefore marks it as `1 -> 2 in progress`, not fully phase 2.

## Recommendation

Do not take another reasoning-core code slice from the stale PR 4-7 backlog.
Semantic cache keys, pack registry, per-review enrichment pack split,
phrase-metadata utility, manifest smoke, graph node contracts, graph routing,
and reflection port adapters have all landed for the current boundary.

The next reasoning-core slice should name a concrete runtime need, such as:

1. a non-Atlas product needs a reusable host adapter contract for LLM/workload
   resolution;
2. a product asks for a slimmed state model beyond the current core state;
3. AI Content Ops needs a new stable provider port or capability check after a
   real generated-asset run exposes an operator/runtime gap.

Do not keep extracting Atlas graph wrappers unless a product-specific runtime
needs one of the stop-rule capabilities above.

## Recommendation History

- 2026-05-17 original audit: take the per-review enrichment pack split next.
- 2026-05-17 post-#564 refresh: the split is closed for the current Atlas
  wrapper boundary. Do not take another slice from the old backlog unless a
  concrete runtime need appears.

## Refresh Log

### 2026-05-17

- Marked the per-review enrichment pack split as closed after verifying
  `atlas_brain.reasoning.review_enrichment` and the atlas evidence-engine
  wrapper tests exist.
- Reframed the next recommendation around concrete runtime triggers instead of
  stale PR-C backlog items.

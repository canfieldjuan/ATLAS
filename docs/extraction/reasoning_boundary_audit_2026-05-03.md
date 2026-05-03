# Reasoning Boundary Audit

Date: 2026-05-03

## Executive Decision

Reasoning should be extracted as its own product boundary:
`extracted_reasoning_core`.

The current extraction shape is already drifting. Content pipeline, competitive
intelligence, LLM infrastructure, and Atlas each own or bridge pieces of the
same reasoning surface. That creates two bad outcomes:

- products lose advanced reasoning when extracted because only leaves were
  copied
- duplicated leaves become forks with different behavior

The fix is not to copy all of `atlas_brain/reasoning/` into every product. The
fix is a shared reasoning core with product-specific reasoning packs.

## Verified Current State

The Atlas reasoning surface contains:

- 30 top-level Python modules
- 7 single-pass prompt Python modules plus prompt package init
- 3 markdown skill prompts
- 1 evidence map YAML
- 9,566 total lines across the files counted above

Duplicated/drifted files checked in this audit:

| File | competitive_intelligence | content_pipeline | llm_infra | Atlas | Status |
| --- | --- | --- | --- | --- | --- |
| `archetypes.py` | no | yes | no | yes | drifted |
| `cross_vendor_selection.py` | yes | no | no | yes | copied/pack candidate |
| `evidence_engine.py` | no | yes | no | yes | drifted |
| `semantic_cache.py` | yes | no | yes | yes | LLM copy matches Atlas, comp-intel is a bridge |
| `temporal.py` | no | yes | no | yes | drifted |
| `wedge_registry.py` | yes | yes | no | yes | drifted/bridged |
| `single_pass_prompts/battle_card_reasoning.py` | yes | no | no | yes | product pack |
| `single_pass_prompts/cross_vendor_battle.py` | yes | no | no | yes | product pack |

Uncloaked direct Atlas dependency:

`extracted_competitive_intelligence/autonomous/tasks/b2b_battle_cards.py`
imports `atlas_brain.reasoning.ecosystem.EcosystemAnalyzer` directly. That
bypasses the bridge pattern and will break standalone competitive-intelligence
imports.

## Boundary Model

Use three layers:

1. `extracted_reasoning_core`
   Shared engine, state, tiers, evidence rules, temporal reasoning,
   archetypes/wedges, narrative planning, falsification, and public ports.

2. Reasoning packs
   Product-specific policy and prompt scaffolding. Examples:
   competitive battle card pack, cross-vendor pack, campaign/content pack,
   vendor briefing pack.

3. Host adapters
   LLM provider routing, semantic-cache storage, event bus, persistence,
   locks, tracing, auth, and product data repositories.

Products consume the core and one or more packs. They must not import internal
core modules directly.

## Core Versus Pack Decision Rule

A module belongs in shared core when:

- it operates on generic evidence, state, tiers, claims, or reasoning context
- two or more products need the same behavior
- it can be expressed behind ports for LLM/cache/persistence
- its output can be reused across content, competitive intelligence, and Atlas

A module belongs in a reasoning pack when:

- it names a specific artifact type, prompt style, or buyer-facing surface
- it encodes product-specific narrative structure
- it contains output JSON schema for battle cards, blog posts, campaigns, or
  vendor briefings
- it is useful to one product family but not the engine itself

A module belongs in host adapters when:

- it talks to Postgres or product tables directly
- it owns locks, event queues, scheduling, tracing, or settings
- it selects concrete LLM providers or cache backends
- it depends on Atlas auth, DB pools, or runtime config

## Public API Of `extracted_reasoning_core`

This API is the contract. Products should import these entry points, not
internal modules.

Stable types:

```python
ReasoningDepth = Literal["L1", "L2", "L3", "L4", "L5"]

@dataclass(frozen=True)
class EvidenceItem:
    source_type: str
    source_id: str
    text: str = ""
    metrics: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class ReasoningInput:
    entity_id: str
    entity_type: str
    goal: str
    evidence: Sequence[EvidenceItem]
    context: Mapping[str, Any] = field(default_factory=dict)
    pack_name: str | None = None

@dataclass(frozen=True)
class ReasoningResult:
    summary: str
    claims: Sequence[Mapping[str, Any]]
    confidence: float
    tier: str
    state: Mapping[str, Any]
    trace: Mapping[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class ReasoningPorts:
    llm: LLMClient | None = None
    semantic_cache: SemanticCacheStore | None = None
    state_store: ReasoningStateStore | None = None
    clock: Clock | None = None
```

Stable entry points:

```python
def validate_wedge(value: str) -> Wedge | None: ...
def get_wedge_meta(wedge: Wedge) -> WedgeMeta: ...
def score_archetypes(
    evidence: Mapping[str, Any],
    temporal: Mapping[str, Any] | None = None,
    *,
    limit: int = 3,
) -> Sequence[ArchetypeMatch]: ...
def evaluate_evidence(
    evidence: Mapping[str, Any],
    *,
    policy: EvidencePolicy | None = None,
) -> EvidenceDecision: ...
def build_temporal_evidence(
    snapshots: Sequence[Mapping[str, Any]],
    *,
    baselines: Mapping[str, Any] | None = None,
) -> TemporalEvidence: ...
def build_narrative_plan(
    context: Mapping[str, Any],
    *,
    pack: ReasoningPack,
) -> NarrativePlan: ...
async def run_reasoning(
    input: ReasoningInput,
    *,
    depth: ReasoningDepth = "L2",
    pack: ReasoningPack | None = None,
    ports: ReasoningPorts | None = None,
) -> ReasoningResult: ...
async def continue_reasoning(
    state: Mapping[str, Any],
    event: Mapping[str, Any],
    *,
    ports: ReasoningPorts | None = None,
) -> ReasoningResult: ...
async def check_falsification(
    claim: Mapping[str, Any],
    fresh_evidence: Sequence[EvidenceItem],
    *,
    policy: FalsificationPolicy | None = None,
    ports: ReasoningPorts | None = None,
) -> FalsificationResult: ...
def compute_evidence_hash(evidence: Mapping[str, Any]) -> str: ...
def build_semantic_cache_key(
    input: ReasoningInput,
    *,
    tier: str,
    pack_name: str | None = None,
) -> str: ...
def load_reasoning_pack(name: str) -> ReasoningPack: ...
def validate_reasoning_output(
    result: ReasoningResult,
    *,
    policy: OutputPolicy | None = None,
) -> ValidationReport: ...
```

Deprecation policy:

- Products may import only from `extracted_reasoning_core` public modules.
- Internal modules may change without deprecation.
- Public entry points need one compatibility shim or migration note before
  removal.
- Reasoning packs can version prompt contracts independently of core.

## Module Classification

| Atlas file | Lines | Classification | Extraction decision |
| --- | ---: | --- | --- |
| `__init__.py` | 15 | package surface | replace with public API exports |
| `agent.py` | 151 | host adapter + graph wrapper | split; core runner behind ports, Atlas tracing/settings in adapter |
| `archetypes.py` | 592 | shared core | consolidate first wave |
| `config.py` | 217 | host adapter/config | define product config dataclasses; Atlas settings adapter later |
| `consumer.py` | 124 | host adapter | leave out of core; adapter over event bus |
| `context_aggregator.py` | 350 | shared core after porting | extract as context builder with data-fetch ports |
| `cross_vendor_selection.py` | 403 | reasoning pack | competitive/cross-vendor pack |
| `ecosystem.py` | 365 | shared core or competitive pack | classify during code audit; direct comp-intel import must be fixed |
| `entity_locks.py` | 244 | persistence adapter | core exposes lock port only |
| `event_bus.py` | 153 | persistence adapter | host adapter |
| `events.py` | 109 | shared core types | extract event dataclasses/envelopes |
| `evidence_engine.py` | 548 | shared core | consolidate first wave |
| `evidence_map.yaml` | 284 | shared core policy data | package as default policy map |
| `falsification.py` | 318 | opinionated shared core | extract with extension points and ports |
| `graph.py` | 652 | shared engine after porting | extract after public API skeleton |
| `graph_prompts.py` | 122 | reasoning pack | pack-specific prompt scaffolding |
| `knowledge_graph.py` | 744 | shared core after porting | extract after state/ports are defined |
| `llm_utils.py` | 131 | LLM port helper | move generic parsing to core; provider calls stay outside |
| `lock_integration.py` | 119 | host adapter | leave out of core |
| `market_pulse.py` | 149 | shared core or market pack | classify with trigger/event pass |
| `narrative.py` | 492 | opinionated shared core | extract with pack extension points |
| `patterns.py` | 237 | shared core | extract after graph/state skeleton |
| `producers.py` | 34 | host adapter | leave out of core; event producer port |
| `prompts.py` | 15 | reasoning pack | replace with pack registry |
| `reflection.py` | 163 | shared core after porting | depends on graph/LLM port |
| `semantic_cache.py` | 338 | split | storage to LLM infra; key derivation/types to core |
| `state.py` | 60 | shared core | extract in skeleton PR |
| `temporal.py` | 490 | shared core | consolidate first wave |
| `tiers.py` | 190 | shared core | extract in skeleton PR |
| `trigger_events.py` | 423 | shared core | extract after temporal/archetype consolidation |
| `wedge_registry.py` | 159 | shared core | first code PR |
| `single_pass_prompts/__init__.py` | 1 | package surface | pack registry |
| `single_pass_prompts/battle_card_reasoning.py` | 311 | reasoning pack | competitive/battle-card pack |
| `single_pass_prompts/category_council_synthesis.py` | 64 | reasoning pack | cross-vendor/category pack |
| `single_pass_prompts/cross_vendor_battle.py` | 55 | reasoning pack | competitive/battle-card pack |
| `single_pass_prompts/cross_vendor_battle_synthesis.py` | 70 | reasoning pack | competitive/cross-vendor pack |
| `single_pass_prompts/reasoning_synthesis.py` | 302 | reasoning pack | general synthesis pack |
| `single_pass_prompts/resource_asymmetry_synthesis.py` | 63 | reasoning pack | competitive/cross-vendor pack |
| `single_pass_prompts/vendor_classify.py` | 174 | reasoning pack | vendor classification pack |
| `skill_prompts/reasoning_analysis.md` | 47 | reasoning pack | prompt pack asset |
| `skill_prompts/reasoning_reflection.md` | 46 | reasoning pack | prompt pack asset |
| `skill_prompts/reasoning_triage.md` | 42 | reasoning pack | prompt pack asset |

## Semantic Cache Split

`semantic_cache.py` currently mixes two concerns:

- cache key/evidence identity and confidence decay
- Postgres-backed storage for `reasoning_semantic_cache`

Split into:

- `extracted_reasoning_core.semantic_cache_keys`
  - `compute_evidence_hash`
  - `build_semantic_cache_key`
  - cache entry dataclasses
  - decay/effective-confidence helpers
- `extracted_llm_infrastructure.reasoning.semantic_cache_store`
  - Postgres implementation
  - lookup/store/validate/invalidate queries
  - migration ownership for `reasoning_semantic_cache`

Reasoning core should use a `SemanticCacheStore` port. It should never import
the LLM infrastructure implementation directly.

## Falsification And Narrative

`falsification.py` and `narrative.py` are more opinionated than
`wedge_registry.py`, `archetypes.py`, or `temporal.py`, but they still belong in
core.

Reason: every product that emits reasoned output needs:

- claims linked to evidence
- falsification conditions
- uncertainty sources
- confidence and support labels
- narrative planning that can be rendered differently per product

Constraint: they need extension points. Product packs should define claim
types, narrative sections, and falsification policies. Core should provide the
runner and validation mechanics.

## Opt-In Policy For Product Use

Reasoning state is opt-in per content type.

Heavy reasoning should be used for:

- long-form reports
- migration guides
- competitive battle cards
- vendor intelligence briefings
- complex multi-section campaign strategy
- 10k+ token narratives that need state, continuity, and falsification

Heavy reasoning should not be mandatory for:

- one-line subject lines
- short social snippets
- simple single-email drafts
- deterministic summaries where the source context is already compact

Products choose a reasoning depth per surface. The content pipeline should
support both simple one-shot generation and stateful long-form reasoning.

## Follow-Up PR Sequence

### PR 1: Boundary Audit

This document.

Acceptance criteria:

- module-by-module classification exists
- public API contract exists
- semantic-cache split is recorded
- opt-in policy exists
- follow-up code sequence is explicit

### PR 2: Core Skeleton + Wedge Consolidation

Create `extracted_reasoning_core/` with:

- public `api.py` / `types.py` / `ports.py`
- `state.py`
- `tiers.py`
- consolidated `wedge_registry.py`
- compatibility imports in content and competitive intelligence

Acceptance criteria:

- `extracted_content_pipeline` and `extracted_competitive_intelligence` consume
  the same wedge registry
- no product owns a forked `wedge_registry.py`
- public API import smoke exists

### PR 3: Semantic Cache Split

Move key derivation and cache-entry types to reasoning core. Keep storage in
LLM infrastructure behind a port.

Acceptance criteria:

- `compute_evidence_hash` has one implementation
- LLM infrastructure owns Postgres queries
- competitive intelligence no longer bridges Atlas semantic cache

### PR 4: Evidence, Temporal, Archetypes Consolidation

Move `evidence_engine.py`, `evidence_map.yaml`, `temporal.py`, and
`archetypes.py` into core.

Acceptance criteria:

- content pipeline imports these from reasoning core
- Atlas can adapt to the shared core without behavior drift
- drifted content copies are removed or converted to wrappers

### PR 5: Reasoning Pack Registry

Extract prompt/policy packs:

- battle card reasoning
- cross-vendor battle
- vendor classify
- reasoning synthesis
- content/campaign pack placeholders

Acceptance criteria:

- packs are explicit dependencies
- core can run without importing a product pack
- products select packs by name/version

### PR 6: Graph/State Engine With Ports

Port `graph.py`, `agent.py`, `context_aggregator.py`, `reflection.py`, and
state progression into the core runner.

Acceptance criteria:

- LLM, cache, state store, event sink, and trace sink are ports
- Atlas-specific tracing/settings/event bus stay in adapters
- products can request depth `L1` through `L5`

### PR 7: Product Migration Pass

Update extracted products to depend on reasoning core:

- content pipeline opts into long-form reasoning for selected surfaces
- competitive intelligence removes direct `atlas_brain.reasoning.ecosystem`
  import
- LLM infrastructure exposes cache store adapter only

Acceptance criteria:

- no runtime `atlas_brain.reasoning` imports in extracted products
- duplicate reasoning leaves are gone
- CI includes drift guard for reasoning-core files

## Immediate Next Code Slice

Start with PR 2, not the full engine.

Scope:

- create `extracted_reasoning_core`
- define public API skeleton and ports
- move/consolidate `wedge_registry.py`
- update content and competitive intelligence to import the shared wedge API
- add import and drift-guard tests

This validates the boundary on the smallest drifted file before touching the
larger engine modules.

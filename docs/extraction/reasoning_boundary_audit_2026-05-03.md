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
    tier: ReasoningDepth
    state: Mapping[str, Any]
    trace: Mapping[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class ReasoningPorts:
    llm: LLMClient | None = None
    semantic_cache: SemanticCacheStore | None = None
    state_store: ReasoningStateStore | None = None
    clock: Clock | None = None
```

Supporting types defined in PR 2:

| Type | Meaning |
| --- | --- |
| `Wedge` | Stable enum of sales/reasoning wedge identifiers shared by prompts, validation, and product renderers. |
| `WedgeMeta` | Label, archetype mapping, sales motion, and required evidence pools for a wedge. |
| `ArchetypeMatch` | One scored archetype result with id, label, score, evidence hits, missing evidence, and risk label. |
| `EvidencePolicy` | Rule set for evidence thresholds, required pools, confidence labels, and section suppression. |
| `EvidenceDecision` | Output of evidence evaluation: allowed/suppressed, confidence, reasons, missing evidence, and trace data. |
| `TemporalEvidence` | Normalized velocity, trend, anomaly, recency, and baseline-relative evidence for a time series. |
| `NarrativePlan` | Product-neutral outline of claims, sections, evidence requirements, and continuity/state hints. |
| `ReasoningPack` | Product-specific prompt and policy bundle selected by name/version. |
| `FalsificationPolicy` | Product or pack rules for deciding which fresh evidence can invalidate a prior claim. |
| `FalsificationResult` | Triggered and non-triggered falsification conditions plus invalidation recommendation. |
| `OutputPolicy` | Validation policy for reasoned outputs, including required claims, citations, confidence, and blocked phrasing. |
| `ValidationReport` | Structured validation result with explicit `passed` verdict, blockers, warnings, repaired fields, and audit trace. |
| `LLMClient` | Port for chat/completion calls; provider routing stays outside reasoning core. |
| `SemanticCacheStore` | Port for lookup/store/validate/invalidate of semantic-cache entries. |
| `ReasoningStateStore` | Port for reading/writing long-running reasoning state and continuation checkpoints. |
| `Clock` | Deterministic time source for recency, cache decay, schedule, and testability. |

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
    reasoning_input: ReasoningInput,
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
    reasoning_input: ReasoningInput,
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

Graph prompt policy:

- `run_reasoning(reasoning_input, ports=ports)` without a product pack is valid.
- Core ships a minimal default pack for generic triage, synthesis, and
  validation.
- Product packs override or extend default graph prompts for battle cards,
  campaigns, vendor briefings, long-form stories, and cross-vendor analysis.
- The graph engine must not import product packs directly; pack selection goes
  through `load_reasoning_pack(name)` or an injected `ReasoningPack`.

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
| `graph_prompts.py` | 122 | default pack + pack override surface | core ships minimal defaults; products override through packs |
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
| `single_pass_prompts/reasoning_synthesis.py` | 302 | default synthesis pack | explicit default pack asset, not miscellaneous core internals |
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
- tests for consolidated `wedge_registry.py` move to
  `tests/test_extracted_reasoning_core_wedge_registry.py`; existing Atlas tests
  remain until their modules migrate

### PR 3: Evidence, Temporal, Archetypes Consolidation

Move `evidence_engine.py`, `evidence_map.yaml`, `temporal.py`, and
`archetypes.py` into core so semantic-cache keys can depend on stable evidence
types instead of today's Atlas-shaped dicts.

Acceptance criteria:

- content pipeline imports these from reasoning core
- Atlas can adapt to the shared core without behavior drift
- drifted content copies are removed or converted to wrappers
- reasoning-core tests own the consolidated evidence, temporal, and archetype
  contracts

### PR 4: Semantic Cache Split

Move key derivation and cache-entry types to reasoning core after the evidence
types settle. Keep storage in LLM infrastructure behind a port.

Acceptance criteria:

- `compute_evidence_hash` has one implementation
- LLM infrastructure owns Postgres queries
- competitive intelligence no longer bridges Atlas semantic cache
- cache tests cover core key derivation separately from the LLM-infra store

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
- CI includes an import-boundary drift guard: extracted products may import
  only `extracted_reasoning_core.api`, `extracted_reasoning_core.types`,
  `extracted_reasoning_core.ports`, or approved pack entry points. Direct
  imports from `extracted_reasoning_core._internal`, concrete module files, or
  `atlas_brain.reasoning` fail CI.

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

---

## PR-C1 Implementation Outcomes (2026-05-04)

This section records the actual outcomes from the PR-C1 sequence (PR-C1a → PR-C1k) so the audit reflects what shipped, not just what was planned. The original PR 2 / PR 3 acceptance criteria from the "Follow-Up PR Sequence" above are now satisfied; the PR 4 / PR 5 / PR 6 / PR 7 sequence remains as planned.

### Slices that landed

| Slice | PR | What it shipped |
| --- | --- | --- |
| PR-C1a | [#94](https://github.com/canfieldjuan/ATLAS/pull/94) | Archetypes consolidation: 10 canonical archetypes in `extracted_reasoning_core/archetypes.py`, frozen `ArchetypeProfile` / `SignalRule` dataclasses, `_ArchetypeMatchInternal` (rich) + `_to_public_match` adapter to public `ArchetypeMatch` |
| PR-C1b | [#100](https://github.com/canfieldjuan/ATLAS/pull/100) | Temporal consolidation: atlas-canonical `TemporalEngine` + content_pipeline's defensive helpers (`_numeric_value` / `_row_get`); canonicalized `MIN_DAYS_FOR_PERCENTILES = 3` (atlas's actual runtime value, not the dead module-constant `= 7`); parameterized `min_days_for_percentiles` constructor knob |
| PR-C1c | [#102](https://github.com/canfieldjuan/ATLAS/pull/102) | Public types promotion: rich `TemporalEvidence` (frozen, slots) + 4 sub-types (`VendorVelocity`, `LongTermTrend`, `CategoryPercentile`, `AnomalyScore`); `ConclusionResult` and `SuppressionResult` |
| PR-C1d | [#104](https://github.com/canfieldjuan/ATLAS/pull/104) | Slim `EvidenceEngine` core: conclusions + suppression surface only; per-review enrichment stays atlas-side until PR 5; both `evaluate_conclusions` (plural) and `evaluate_conclusion` (singular) shipped per audit Collision 4 resolution |
| PR-C1g | [#103](https://github.com/canfieldjuan/ATLAS/pull/103) | API wiring: `score_archetypes`, `build_temporal_evidence`, `evaluate_evidence` stubs from PR-C1d wired through to the consolidated engines |
| PR-C1h | [#111](https://github.com/canfieldjuan/ATLAS/pull/111) | `extracted_content_pipeline/reasoning/archetypes.py`: 590-line drifted fork → 44-line wrapper. Drift-forward: `_numeric_value` into core's `_evaluate_rule` to handle messy string values |
| PR-C1i | [#117](https://github.com/canfieldjuan/ATLAS/pull/117) | `extracted_content_pipeline/reasoning/evidence_engine.py`: 338-line drifted fork → 85-line wrapper. Drift-forwards into core: `EvidenceEngine.from_rules(dict)` classmethod (in-memory construction), lazy yaml import + JSON suffix detection, `_numeric_value` in numeric checks, `min_count`/`exists` operator parity, dual-form suppression. Wrapper carries content-pipeline-specific `_DEFAULT_RULES` dict (consumer-review conclusions: `pricing_crisis`, `losing_market_share`, `active_churn_wave`, `support_quality_risk`) |
| PR-C1j | [#127](https://github.com/canfieldjuan/ATLAS/pull/127) | `extracted_content_pipeline/reasoning/temporal.py`: 466-line drifted fork → 49-line wrapper. Drift-forwards into core: `_coerce_date`, `_days_between`, `_volatility`, `_percentiles_from_rows` helpers; vendor_name `.strip()` normalization; self-contained `_compute_percentiles` SELECT (drops atlas import). Latent bug fix: `analyze_vendor` and `_compute_long_term_trends` were mutating frozen `TemporalEvidence` / `LongTermTrend` (activated by PR-C1c freeze) -- fixed by constructing dataclasses with all fields at once |
| PR-C1k | [#134](https://github.com/canfieldjuan/ATLAS/pull/134) | Test rename: `tests/test_extracted_reasoning_{archetypes,evidence_engine,temporal}.py` → `tests/test_extracted_content_pipeline_reasoning_*.py`; audit-doc amendment to `evidence_temporal_archetypes_audit_2026-05-03.md` |

### Architectural deviations from the original plan

Two judgment calls during implementation departed from the audit's original PR 3 acceptance criteria. Both are documented in the per-slice doc / PR description; surfaced here for the audit trail.

1. **Atlas adaptation deferred to PR 5/6** -- the original PR 3 line "Atlas can adapt to the shared core without behavior drift" turned out to require an atlas-side migration that didn't fit the consolidation slice. Atlas continues to use its own `atlas_brain/reasoning/temporal.py` and `atlas_brain/reasoning/archetypes.py` (forks of the now-canonical core). The content_pipeline mirrors are routed through core; the atlas-side migration is part of the PR 7 "Product Migration Pass" sequence, not PR-C1. Core's `TemporalEngine` had no production callers in atlas during PR-C1, which is why latent bugs (frozen-dataclass mutations) only surfaced once content_pipeline routed through it.

2. **Test rename deviation** -- the original PR 3 acceptance criterion "reasoning-core tests own the consolidated evidence, temporal, and archetype contracts" was interpreted in the per-slice audit (`evidence_temporal_archetypes_audit_2026-05-03.md`) as "rename `tests/test_extracted_reasoning_*.py` and redirect imports to `extracted_reasoning_core.*`". That plan turned out to be wrong for `evidence_engine` because the wrapper carries content-pipeline-specific `_DEFAULT_RULES` (consumer-review conclusions absent from core's `evidence_map.yaml`). Redirecting imports would have broken the assertions. PR-C1k instead renamed the files to `test_extracted_content_pipeline_reasoning_*.py` and kept imports unchanged; the canonical core tests live separately at `test_extracted_reasoning_core_*.py` (unit-style). Both layers are kept; coverage is complementary. Per-slice audit (`evidence_temporal_archetypes_audit_2026-05-03.md`) carries the detailed deviation note; the original "redirect" table is preserved with `~~SUPERSEDED~~` strikethrough markers.

### What did not ship in PR-C1

The following remain on the PR 4 / PR 5 / PR 6 / PR 7 backlog as originally planned:

- **PR 4: Semantic cache split.** Not touched in PR-C1.
- **PR 5: Reasoning pack registry.** Per-review enrichment (`compute_urgency`, `override_pain`, `derive_recommend`, etc.) still lives in atlas-side `atlas_brain/reasoning/` and has not been carved into a pack module. This was an explicit scope-out in PR-C1d's slim-core split.
- **PR 6: Graph/state engine with ports.** Not touched in PR-C1.
- **PR 7: Product migration pass.** Atlas-side reasoning still uses the atlas-local forks of `archetypes.py` / `temporal.py` / `evidence_engine.py`. Removing those forks (and pointing atlas at core) is the PR 7 work.

### Drift-forward summary

PR-C1 surfaced a recurring pattern: surface comparison of "what's in core vs what's in the fork" missed defensive helpers that only manifested under test. The slim core was the better-engineered _interface_, but the content_pipeline fork carried better defensive coercion in helpers (`_numeric_value`, `_row_get`, `_coerce_date`, `_days_between`, `_volatility`, `_percentiles_from_rows`) and more tolerant operator handling (`min_count`, `exists`, dual-form suppression). Each wrapper-conversion slice (PR-C1h / PR-C1i / PR-C1j) carried these forward into core in the same commit as the wrapper. Future product-migration slices (PR 7) should expect the same pattern when atlas-side reasoning is finally pointed at core.

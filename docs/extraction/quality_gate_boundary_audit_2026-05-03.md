# Quality Gate Extraction Boundary Audit

Date: 2026-05-03
Owner: codex-2026-05-03
Coordination slice: PR-B1
Status: Audit proposal, docs only

## Summary

`extracted_quality_gate` should be a reusable gate kernel plus optional packs, not a dump of every Atlas module with "quality" in the name.

The reusable product is an AI output governance layer:

- normalize claims and evidence into a stable envelope
- evaluate deterministic gate policies
- return structured pass/warn/block decisions
- expose ports for approvals, audit logs, evidence stores, clocks, and embedding similarity
- let products provide their own packs for blog, campaign, witness, scrape-source, and B2B evidence behavior

The first code PR should start with `atlas_brain/services/b2b/product_claim.py`. It is already the cleanest shared contract: pure, deterministic, pool-free, and explicitly designed as the envelope consumed by reports, UI cards, and downstream renderers.

`safety_gate.py` is the second-best candidate, but it must be split. Its content/risk decisions are reusable; DB-backed approvals and event logging are Atlas adapters.

The blog, campaign, witness, evidence, scrape, and synthesis validators should not move into core first. They are valuable, but they are product packs or Atlas adapters over the core gate contract.

## Product Thesis

The standalone module should answer one question consistently:

> Given content, evidence, claims, and policy, is this safe and specific enough to render, send, publish, or hand to another system?

This unlocks sellable products without hard-coding Atlas:

- AI content QA before publishing
- outbound campaign quality and evidence coverage checks
- customer-facing quote and witness rendering policy
- product claim confidence and suppression gates
- safety/risk review and approval workflows
- memory or conversation quality signals
- source-quality checks for ingestion pipelines

The key boundary is that `extracted_quality_gate` makes gate decisions. It does not own scraping, content generation, reasoning orchestration, LLM provider routing, database tables, or customer-specific rendering.

## Decision Rules

A module belongs in shared core when it:

- is deterministic or accepts explicit ports for external work
- has no Atlas database dependency
- has no task scheduler dependency
- can evaluate a gate without knowing the product that called it
- returns structured decisions instead of rendering customer copy

A module belongs in a product pack when it:

- maps product-specific rows into shared gate inputs
- applies product-specific policies, thresholds, or field names
- knows about blog, campaign, B2B witness, source-quality, or synthesis semantics

A module stays Atlas-only when it:

- runs backfills or maintenance jobs
- directly uses `get_db_pool`, Atlas settings, scheduler tasks, or event publishers
- reads or writes Atlas tables
- wires the gate into an API endpoint or autonomous task

## Source Inventory

| Atlas file | Approx. LOC | Classification | Extraction decision |
| --- | ---: | --- | --- |
| `atlas_brain/services/b2b/product_claim.py` | 546 | Shared core candidate | Extract first. This is the cleanest claim envelope and render-gate policy. |
| `atlas_brain/services/safety_gate.py` | 552 | Split core plus ports plus Atlas adapter | Extract content/risk scan and decision model; leave approvals, DB, audit events behind adapters. |
| `atlas_brain/memory/quality.py` | 259 | Optional interaction-quality pack | Split regex correction signals from embedding repetition detection. Embeddings must be a port. |
| `atlas_brain/services/blog_quality.py` | 539 | Blog quality pack | Later pack over the core. Current module reaches into blog task internals and B2B specificity helpers. |
| `atlas_brain/services/campaign_quality.py` | 556 | Campaign quality pack | Later pack over the core. Current module depends on settings, specificity, and reasoning-view loading. |
| `atlas_brain/services/b2b/witness_render_gate.py` | 161 | B2B witness adapter | Product pack. It maps B2B witness rows onto product-claim gates. |
| `atlas_brain/services/b2b/evidence_gate.py` | 124 | B2B evidence coverage adapter | Product pack or Atlas adapter. It is DB-pool based and campaign specific. |
| `atlas_brain/services/b2b/evidence_claim.py` | 622 | Evidence-claim policy pack | Not initial core. Useful policy logic, but B2B evidence semantics need a pack boundary. |
| `atlas_brain/services/b2b/evidence_claim_builder.py` | 409 | Evidence-claim producer adapter | Atlas/B2B adapter. It writes claim attempts and replay state. |
| `atlas_brain/services/witness_quality_propagation.py` | 121 | Row decoration adapter | Adapter. Useful after core exists, but not a standalone gate kernel. |
| `atlas_brain/services/witness_quality_maintenance.py` | 325 | Maintenance tooling | Atlas-only. Backfill and propagation jobs should not enter core. |
| `atlas_brain/autonomous/tasks/_b2b_synthesis_validation.py` | 1663 | Synthesis validation pack | Product pack. Keep out of first wave; likely bridges reasoning core and quality gate later. |
| `atlas_brain/autonomous/tasks/b2b_scrape_intake.py` | 2973 | Source-quality ingest adapter | Extract only source-quality helpers later; task orchestration stays Atlas-only. |
| `atlas_brain/services/blog_quality_backfill.py` | unknown | Backfill tooling | Atlas-only. |
| `atlas_brain/services/blog_visibility_backfill.py` | unknown | Backfill tooling | Atlas-only. |
| `atlas_brain/api/blog_admin.py` | unknown | API consumer | Atlas-only wiring. |
| `atlas_brain/api/b2b_campaigns.py` | unknown | API consumer | Atlas-only wiring. |
| `atlas_brain/api/b2b_scrape.py` | unknown | API consumer | Atlas-only wiring. |
| `atlas_brain/autonomous/tasks/b2b_battle_cards.py` | large | Product consumer | Do not extract as core. Keep as product pack or adapter after reasoning boundaries land. |
| `atlas_brain/autonomous/tasks/b2b_campaign_generation.py` | large | Product consumer | Do not extract as core. Quality gate should validate campaign artifacts, not own generation. |

## Core Candidate: Product Claim Contract

`atlas_brain/services/b2b/product_claim.py` is the strongest first extraction because it already behaves like a public contract.

Current reusable pieces:

- `ClaimScope`
- `EvidencePosture`
- `ConfidenceLabel`
- `SuppressionReason`
- `ClaimGatePolicy`
- `derive_evidence_posture`
- `derive_confidence`
- `decide_render_gates`
- `compute_claim_id`
- `ProductClaim`
- `build_product_claim`

Why it is core:

- deterministic
- no DB pool
- no scheduler dependency
- no LLM dependency
- converts evidence and confidence into structured render decisions
- already describes itself as the shared envelope for UI cards and report sections

What to avoid:

- Do not preserve the `b2b` namespace in the standalone public API.
- Do not force every product into B2B field names.
- Do not let product packs import internal claim helpers directly; expose the stable contract through `extracted_quality_gate.api` and `extracted_quality_gate.types`.

## Split Candidate: Safety Gate

`atlas_brain/services/safety_gate.py` has two different modules inside one file.

Reusable core:

- deterministic content checks
- risk-level assessment
- gate decision construction
- policy thresholds and categories

Atlas adapter:

- intervention approval persistence
- pending approval listing
- approval status checks
- event logging through Atlas events
- `get_db_pool`
- settings-derived runtime behavior

Recommended split:

- `extracted_quality_gate.safety` for pure scan and risk decisions
- `extracted_quality_gate.ports.ApprovalStore`
- `extracted_quality_gate.ports.AuditLog`
- `atlas_brain` wrapper that binds those ports to current DB tables and event logging

## Product Packs

### Blog Quality Pack

`atlas_brain/services/blog_quality.py` should become a blog pack, not core.

It currently does useful work:

- latest audit lookup
- failure explanation projection
- revalidation context construction
- row-to-blueprint projection

But it depends on Atlas-specific blog rows, B2B specificity helpers, and task internals from the blog generator. The shared core should provide reports and gate primitives; the blog pack should translate blog rows into those primitives.

### Campaign Quality Pack

`atlas_brain/services/campaign_quality.py` should become a campaign pack.

It currently does useful work:

- campaign specificity context
- campaign revalidation
- async fallback revalidation
- reasoning context handoff

But it depends on Atlas settings, specificity helpers, and campaign-specific reasoning view loading. The standalone quality gate should not own those dependencies.

### Witness And Render Policy Pack

`atlas_brain/services/b2b/witness_render_gate.py` is an adapter over the product claim contract.

It should stay product-specific because it knows:

- witness row shape
- customer-visible render gates
- B2B confidence and evidence fields
- phrase and witness semantics

The pack should import the public claim API, not internal product-claim helpers.

### Evidence Claim Coverage Pack

`atlas_brain/services/b2b/evidence_gate.py`, `evidence_claim.py`, and `evidence_claim_builder.py` are related but not identical.

Recommended split:

- policy types and deterministic claim coverage checks can become a B2B evidence pack
- DB query and write orchestration stays Atlas-only
- shadow/replay behavior stays in Atlas until a standalone store port is defined

Do not fold these into the core kernel in the first pass. They carry real product semantics.

### Source Quality Pack

`b2b_scrape_intake.py` contains source-quality gate helpers such as `_quality_gate_skip_reason` and `_should_apply_source_quality_gate`.

Those helpers are valuable, but the file is a large ingestion task. Extracting the task would pull scheduler, source, vendor, scrape, and storage concerns into the quality module.

Recommended approach:

- later extract a small `source_quality` pack with pure rules
- leave scrape orchestration and persistence in Atlas

### Memory Quality Pack

`atlas_brain/memory/quality.py` has reusable quality-signal ideas:

- correction detection
- contradiction signals
- repetition detection

But repetition detection depends on embeddings and in-memory similarity state. If extracted, embeddings must be a port:

- `EmbeddingSimilarity`
- `MemoryQualityStore` if persistence is introduced

This pack is useful, but not first wave.

## Public API Proposal

The first code PR should expose the smallest useful surface and keep internals private.

```python
from extracted_quality_gate.api import (
    assess_risk,
    build_product_claim,
    check_content_safety,
    decide_render_gates,
    derive_confidence,
    derive_evidence_posture,
    evaluate_quality,
    project_quality_failure,
    run_gate,
)
```

Stable types:

```python
from dataclasses import dataclass, field as dataclass_field

class GateSeverity(str, Enum): ...
class GateDecision(str, Enum): ...
class ClaimScope(str, Enum): ...
class EvidencePosture(str, Enum): ...
class ConfidenceLabel(str, Enum): ...
class SuppressionReason(str, Enum): ...

@dataclass(frozen=True)
class GateFinding:
    code: str
    message: str
    severity: GateSeverity
    field: str | None = None
    metadata: Mapping[str, Any] = dataclass_field(default_factory=dict)

@dataclass(frozen=True)
class QualityReport:
    passed: bool
    decision: GateDecision
    findings: tuple[GateFinding, ...] = ()
    blockers: tuple[GateFinding, ...] = ()
    warnings: tuple[GateFinding, ...] = ()
    metadata: Mapping[str, Any] = dataclass_field(default_factory=dict)

@dataclass(frozen=True)
class ClaimGatePolicy:
    min_confidence: ConfidenceLabel
    allow_unvalidated: bool = False
    allow_conflicting: bool = False

@dataclass(frozen=True)
class ProductClaim:
    claim_id: str
    claim_type: str
    scope: ClaimScope
    text: str
    evidence_posture: EvidencePosture
    confidence: ConfidenceLabel
    render_allowed: bool
    suppression_reasons: tuple[SuppressionReason, ...] = ()
    metadata: Mapping[str, Any] = dataclass_field(default_factory=dict)

@dataclass(frozen=True)
class QualityPolicy:
    name: str
    version: str
    thresholds: Mapping[str, Any] = dataclass_field(default_factory=dict)
    metadata: Mapping[str, Any] = dataclass_field(default_factory=dict)

@dataclass(frozen=True)
class QualityInput:
    artifact_type: str
    artifact_id: str | None
    content: str | None
    evidence: tuple[Mapping[str, Any], ...] = ()
    claims: tuple[Mapping[str, Any], ...] = ()
    context: Mapping[str, Any] = dataclass_field(default_factory=dict)
```

Stable entry points:

```python
def derive_evidence_posture(evidence: Mapping[str, Any], *, policy: ClaimGatePolicy | None = None) -> EvidencePosture: ...

def derive_confidence(evidence: Mapping[str, Any], *, policy: ClaimGatePolicy | None = None) -> ConfidenceLabel: ...

def decide_render_gates(claim: ProductClaim, *, policy: ClaimGatePolicy | None = None) -> QualityReport: ...

def build_product_claim(
    *,
    claim_type: str,
    scope: ClaimScope,
    text: str,
    evidence: Mapping[str, Any],
    policy: ClaimGatePolicy | None = None,
) -> ProductClaim: ...

def check_content_safety(text: str, *, policy: QualityPolicy | None = None) -> QualityReport: ...

def assess_risk(signals: Mapping[str, Any], *, policy: QualityPolicy | None = None) -> QualityReport: ...

def evaluate_quality(subject: QualityInput, *, policy: QualityPolicy, ports: "QualityPorts | None" = None) -> QualityReport: ...

def project_quality_failure(report: QualityReport, *, audience: str = "internal") -> Mapping[str, Any]: ...

async def run_gate(subject: QualityInput, *, policy: QualityPolicy, ports: "QualityPorts | None" = None) -> QualityReport: ...
```

The public API should not expose module internals. Products should import from:

- `extracted_quality_gate.api`
- `extracted_quality_gate.types`
- `extracted_quality_gate.ports`
- explicitly named packs such as `extracted_quality_gate.packs.blog`

## Ports

The core module should define ports but not implement Atlas storage.

```python
class Clock(Protocol):
    def now(self) -> datetime: ...

class AuditLog(Protocol):
    async def record_gate_event(self, event: Mapping[str, Any]) -> None: ...

class ApprovalStore(Protocol):
    async def create_request(self, request: Mapping[str, Any]) -> str: ...
    async def get_status(self, request_id: str) -> Mapping[str, Any] | None: ...

class EvidenceClaimStore(Protocol):
    async def fetch_claims(self, *, artifact_type: str, artifact_id: str) -> Sequence[Mapping[str, Any]]: ...

class EmbeddingSimilarity(Protocol):
    async def similarity(self, left: str, right: str) -> float: ...

class PolicyProvider(Protocol):
    def get_policy(self, name: str, *, version: str | None = None) -> QualityPolicy: ...

@dataclass(frozen=True)
class QualityPorts:
    clock: Clock | None = None
    audit_log: AuditLog | None = None
    approval_store: ApprovalStore | None = None
    evidence_claim_store: EvidenceClaimStore | None = None
    embedding_similarity: EmbeddingSimilarity | None = None
    policy_provider: PolicyProvider | None = None
```

Atlas can bind these ports to:

- Postgres tables
- Atlas settings
- event logs
- approval workflows
- embedding services
- evidence claim repositories

## Relationship To Reasoning Core

Quality gate should not become a reasoning engine.

Reasoning core can produce:

- claims
- evidence bundles
- narrative plans
- validation candidates
- confidence signals

Quality gate consumes those outputs and decides:

- render allowed
- send allowed
- publish allowed
- approval required
- suppression reason
- warning/failure projection

The two products should integrate through stable inputs and outputs, not through internal imports.

## Relationship To LLM Infrastructure

Quality gate should not own provider routing, token accounting, retries, model selection, or cache storage.

Acceptable LLM-infra dependencies are ports:

- embedding similarity for memory/repetition quality
- optional LLM reviewer for a future premium pack, behind an explicit `LLMReviewPort`

The first extraction should be deterministic and runnable without a model key.

## Test Surface

Existing tests that should seed the extracted module:

| Current test file | Migration target |
| --- | --- |
| `tests/test_product_claim_contract.py` | Move or mirror into `extracted_quality_gate/tests/test_product_claim_contract.py` in PR-B2. |
| `tests/test_evidence_gate.py` | Keep Atlas tests now; later move deterministic policy tests into B2B evidence pack. |
| `tests/test_blog_quality_service.py` | Keep Atlas tests now; later convert to blog pack tests. |
| `tests/test_campaign_quality_service.py` | Keep Atlas tests now; later convert to campaign pack tests. |
| `tests/test_memory_quality.py` | Split pure regex tests from embedding-port tests if memory pack is extracted. |
| `tests/test_witness_quality_propagation.py` | Adapter tests; keep Atlas until witness pack extraction. |
| `tests/test_witness_quality_maintenance.py` | Atlas-only backfill tests. |
| `tests/test_b2b_blog_quality_gate.py` | Product integration tests; keep in Atlas or content pipeline. |
| `tests/test_b2b_vendor_briefing_quote_gate.py` | Product integration tests; depends on witness/render policy. |

PR-B2 should include a minimal standalone test contract:

- dataclasses instantiate with defaults
- product claim IDs are deterministic
- evidence posture derivation is stable
- confidence derivation is stable
- render gates suppress unsupported or conflicting claims
- public imports work from `api`, `types`, and `ports`
- no import of `atlas_brain` from core package

## Sequencing Plan

### PR-B1: Boundary Audit

This document.

Acceptance criteria:

- classify the quality-gate surface
- define what is core, pack, adapter, and Atlas-only
- define first public API and ports
- define follow-up PR sequence

### PR-B2: Core Skeleton And Product Claim Contract

Create `extracted_quality_gate`.

Move or copy the deterministic product-claim contract into:

- `extracted_quality_gate/types.py`
- `extracted_quality_gate/api.py`
- `extracted_quality_gate/product_claim.py`
- `extracted_quality_gate/ports.py`

Add tests from `test_product_claim_contract.py`.

Add a compatibility wrapper in Atlas only if needed.

Acceptance criteria:

- package imports without Atlas installed
- tests run without database, settings, or model keys
- products import only public API

### PR-B3: Safety Gate Split

Extract the deterministic part of `safety_gate.py`.

Move Atlas-specific approval and audit behavior behind:

- `ApprovalStore`
- `AuditLog`
- `Clock`

Acceptance criteria:

- content safety scan works standalone
- risk decision works standalone
- Atlas wrapper preserves current behavior

### PR-B4: Blog And Campaign Packs

Create optional packs:

- `extracted_quality_gate.packs.blog`
- `extracted_quality_gate.packs.campaign`

Keep content-generation orchestration in `extracted_content_pipeline` or Atlas.

Acceptance criteria:

- blog and campaign quality reports share the core `QualityReport`
- product-specific field mapping is confined to pack modules
- no pack imports Atlas internals

### PR-B5: B2B Evidence, Witness, And Source Packs

Extract deterministic adapters:

- witness render gate mapping
- evidence claim coverage policy
- source-quality rules from scrape intake

Keep DB queries and backfills in Atlas.

Acceptance criteria:

- B2B packs consume public core API
- Atlas DB adapters are thin wrappers
- customer-facing render gates still use strict evidence markers

### PR-B6: Product Migration And Drift Guard

Migrate products to the standalone package.

Add CI checks:

- standalone package imports without Atlas
- products do not import `extracted_quality_gate` internals
- products do not keep duplicate copies of core files
- public API import smoke tests pass for each extracted product

Preferred drift guard:

- disallow imports from `extracted_quality_gate._internal` or non-public modules
- allow only `api`, `types`, `ports`, and explicit `packs.*`
- compare duplicated files only as a temporary migration aid, not as the long-term guard

## Risks

### Risk: Extracting Too Much

The largest risk is treating every quality-related file as core. That would recreate Atlas inside the standalone product.

Mitigation:

- start with pure product-claim contract
- split safety gate carefully
- force blog, campaign, witness, and source behavior into packs

### Risk: Product Packs Reach Into Internals

If packs import private helpers, the extracted module will drift again.

Mitigation:

- public API first
- import guard in CI
- no product imports from internal modules

### Risk: Atlas Adapters Hide Core Assumptions

Atlas wrappers may preserve current behavior while standalone users hit missing policy defaults.

Mitigation:

- standalone tests must run without Atlas
- default policies must be explicit
- every port has a no-op or in-memory test implementation

### Risk: Reasoning And Quality Boundaries Blur

Synthesis validation can look like reasoning. Reasoning can look like quality. The split should be operational:

- reasoning proposes or explains
- quality gate decides pass, warn, block, suppress, or approval required

## Open Questions

1. Should `ProductClaim` stay as the primary envelope name, or should the standalone package use a more general `Claim` and provide `ProductClaim` as an alias?
2. Should the first package include only deterministic gates, or should an optional LLM-review port be defined early but left unimplemented?
3. Should evidence-claim logic become a B2B pack under quality gate, or live closer to reasoning core as claim validation?
4. Should memory quality be part of this product, or a separate conversation-quality package that depends on quality gate types?
5. Should approval workflows ship as a reference in-memory adapter, or only as interfaces in the first standalone release?

## Recommendation

Proceed with PR-B2 as the first code PR:

1. scaffold `extracted_quality_gate`
2. extract `product_claim.py` into stable public API/types
3. move or mirror the product-claim contract tests
4. add an import smoke test proving the package works without Atlas

Do not start with blog or campaign quality. Those are important, but they depend on product-specific context and will be cleaner once the shared claim and quality-report contracts exist.

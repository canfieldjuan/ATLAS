# Atlas Churn Signals reasoning-engine map

## High-level pipeline

1. **Ingestion/intake**: `b2b_scrape_intake` imports review-like source rows and stages them for enrichment.
2. **Review reasoning (Tiered extraction/classification)**: `b2b_enrichment` runs two-tier LLM + deterministic post-processing and sets each row to `enriched`, `no_signal`, or `quarantined`.
3. **Deterministic aggregation**: `b2b_churn_intelligence` builds churn-signal and intelligence pools (evidence/segment/temporal/displacement/category/account).
4. **Vendor reasoning synthesis**: `b2b_reasoning_synthesis` converts pooled evidence into validated reasoning contracts with witness/source traceability.
5. **Reasoning normalization/reuse layers**:
   - `_b2b_synthesis_reader` provides a typed read contract over v1/v2 synthesis rows.
   - `_b2b_reasoning_contracts` decomposes battle-card-shaped output into reusable contracts.
   - `_b2b_reasoning_atoms` derives deterministic “reasoning atoms” from contracts + witness packets.
6. **Cross-vendor reasoning**: `_b2b_cross_vendor_synthesis` builds deterministic packets for vendor-vs-vendor, council, and asymmetry conclusions.
7. **Downstream product consumption**: battle cards, reports, scorecards, MCP/API/UI read these persisted reasoning artifacts.

## Separate reasoning engines/systems in churn signals

### 1) Tiered review-reasoning engine (per review)
- **Where**: `atlas_brain/autonomous/tasks/b2b_enrichment.py`
- **What it does**:
  - Tier 1 extraction for base churn fields.
  - Tier 2 classification only when Tier 1 has gaps.
  - Deterministic validation/derivation/repair and status assignment (`enriched` / `no_signal` / `quarantined`).
- **Why it matters**: this is the foundational semantic layer; everything downstream assumes this normalized enrichment shape.

### 2) Deterministic pool builder (per vendor/category)
- **Where**: `atlas_brain/autonomous/tasks/b2b_churn_intelligence.py` + shared builders in `_b2b_shared.py`
- **What it does**:
  - Explicitly *does not* run LLM vendor reasoning anymore.
  - Builds/persists canonical pool layers used by synthesis and reports.
- **Why it matters**: this is the structured evidence substrate for later reasoning.

### 3) Vendor reasoning synthesis engine (Stage 5)
- **Where**: `atlas_brain/autonomous/tasks/b2b_reasoning_synthesis.py`
- **What it does**:
  - Consumes pooled data, builds witness-backed packets, calls synthesis LLM.
  - Validates quality (reject/weak/pass).
  - Persists reusable reasoning contracts (`vendor_core_reasoning`, `displacement_reasoning`, `category_reasoning`, `account_reasoning`).
- **Why it matters**: this is the primary reusable “reasoning conclusion” layer.

### 4) Reasoning-contract decomposition engine
- **Where**: `atlas_brain/autonomous/tasks/_b2b_reasoning_contracts.py`
- **What it does**:
  - Normalizes/decomposes synthesis output into stable contract blocks independent of specific report schemas.
- **Why it matters**: allows multiple products to consume one consistent reasoning schema.

### 5) Reasoning-atoms derivation engine
- **Where**: `atlas_brain/autonomous/tasks/_b2b_reasoning_atoms.py`
- **What it does**:
  - Deterministically derives lower-level atom structures and lineage (`metric_ids`, `witness_ids`, evidence freshness) from persisted contracts/packets.
- **Why it matters**: gives explainable, composable “reasoning primitives” for UI/API/product features.

### 6) Cross-vendor reasoning packet engine
- **Where**: `atlas_brain/autonomous/tasks/_b2b_cross_vendor_synthesis.py`
- **What it does**:
  - Builds pairwise/council/asymmetry evidence packets and hashes.
  - Supports persisted cross-vendor conclusions for comparative intelligence.
- **Why it matters**: reusable comparative reasoning separate from single-vendor synthesis.

### 7) Typed reasoning reader contract
- **Where**: `atlas_brain/autonomous/tasks/_b2b_synthesis_reader.py`
- **What it does**:
  - Abstracts v1/v2 synthesis schema differences.
  - Extracts reference IDs, packet artifacts, confidence normalization for downstream consumers.
- **Why it matters**: compatibility layer that prevents each consumer from re-implementing parsing logic.

## Supporting infra these systems depend on

- **DB schema + persistence artifacts**:
  - `b2b_reviews`, `b2b_churn_signals` baseline tables.
  - witness packet tables (`b2b_vendor_reasoning_packets`, `b2b_vendor_witnesses`).
  - evidence-claim contract table (`b2b_evidence_claims`) for validated claim selection and rollout.
- **Shared deterministic builders/helpers**:
  - heavy use of `_b2b_shared.py` readers/aggregators/score builders.
- **LLM pipeline + routing + telemetry**:
  - pipeline LLM clients/routing, tracing, cache metrics used during synthesis.
- **Reasoning registries/utilities**:
  - wedge registry/validation, semantic hashing/cache utilities.
- **Consumer interfaces**:
  - MCP tools (`atlas_brain/mcp/b2b/signals.py`, `.../write_intelligence.py`) read and overlay reasoning into API outputs.

## Re-creating for other use-cases: feasibility + likely missing pieces

### What is portable with minimal changes
- `b2b_reasoning_synthesis` quality-gating pattern.
- `_b2b_synthesis_reader` typed reader abstraction.
- `_b2b_reasoning_contracts` + `_b2b_reasoning_atoms` decomposition strategy.
- cross-vendor packet + evidence-hash approach.

### What is coupled and usually blocks extraction
1. **Domain schema coupling**
   - Current contracts assume churn vocabulary (pain, displacement, migration, wedge types).
2. **SQL/read-model coupling**
   - Pool builders and packet fallbacks read churn-specific tables and columns.
3. **Status-machine coupling**
   - Enrichment states and repair paths are churn-specific.
4. **Prompt/skill coupling**
   - Extraction and synthesis prompts are domain specific.
5. **Consumer-contract coupling**
   - Downstream code expects current contract keys and confidence labels.

### New code usually required for compatibility in another domain
- Define a **new domain ontology** and section-contract schema.
- Build a **domain-specific enrichment normalizer** (or adapters into current intermediate schema).
- Implement domain **pool builders** equivalent to current evidence/segment/temporal/displacement layers.
- Create **packet builders + validators** for your domain’s evidence semantics.
- Add **reader adapters** so existing consumer surfaces (MCP/API/UI) can read the new contracts.
- Add/extend DB migrations for new artifact tables if reusing only part of current schema.

## Practical extraction checklist

1. Start by extracting these modules together as one unit:
   - `b2b_enrichment.py`
   - `b2b_churn_intelligence.py`
   - `b2b_reasoning_synthesis.py`
   - `_b2b_synthesis_reader.py`
   - `_b2b_reasoning_contracts.py`
   - `_b2b_reasoning_atoms.py`
   - `_b2b_cross_vendor_synthesis.py`
2. Pull required table migrations (at minimum):
   - `055_b2b_reviews.sql`
   - `247_b2b_vendor_witness_packets.sql`
   - `305_b2b_evidence_claims.sql`
3. Include shared infra:
   - `_b2b_shared.py`, LLM pipeline/routing/tracing, wedge registry, semantic hash/cache utils.
4. Add a domain adapter layer before touching prompts.

## Recommendation: extract vs rebuild (based on current extracted products)

### Short answer
Use a **hybrid strategy**:
- **Extract and reuse** the existing reasoning substrate modules where Atlas already has stable standalone seams.
- **Rebuild product-specific reasoning producers/contracts** when the destination product has a different ontology or different operational constraints.

### Why this is the best fit in this repo right now

- `extracted_llm_infrastructure` is already at standalone/runtime-decoupled maturity; it is the strongest reusable base for any reasoning product (routing, providers, cache, tracing, cost).  Rebuilding this would duplicate solved plumbing.
- `extracted_competitive_intelligence` is partially standalone but still has explicit Phase-3 decoupling work for deep builders and `_b2b_shared`/task adapters; this indicates the reasoning *consumer* surface is reusable, but full producer extraction remains coupled.
- `extracted_content_pipeline` explicitly treats reasoning generation as host-owned and consumes compressed reasoning via ports/contracts rather than importing synthesis internals; this is a strong pattern for product reuse.

### Decision framework

Choose **extract/reuse existing reasoning module** when all are true:
1. New use case can live with current confidence labels + witness/reference-id semantics.
2. Existing pool layers (or a thin adapter) can feed required facts.
3. Product can consume through typed reader/contract ports.

Choose **rebuild using Atlas pattern** when any are true:
1. Domain ontology differs (claims, wedges, evidence semantics, risk labels).
2. Evidence sources/time windows differ materially from churn-review assumptions.
3. Product needs different governance rules (validation/rejection thresholds, compliance constraints).

### Concrete plan I recommend

1. **Do not extract all churn reasoning engines as one generic package immediately.**
2. **First standardize interfaces**:
   - treat `extracted_llm_infrastructure` as shared substrate,
   - keep reasoning producer behind host ports (like content pipeline’s `CampaignReasoningContextProvider` pattern),
   - keep typed read contracts for downstream consumers.
3. **Then fork/rebuild only domain-specific producer logic** (pool builders, prompts, validators, contract schema) per new product.
4. **Optionally upstream common deterministic utilities** (hashing, lineage/ref IDs, section quality gates) into a small shared reasoning-core library after 2+ products need the same invariant.

This gives fastest delivery with least hidden coupling risk: reuse stable infra, avoid dragging churn-specific SQL/state machines into unrelated products.

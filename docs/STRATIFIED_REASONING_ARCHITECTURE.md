# Stratified Reasoning Architecture

## B2B Churn Intelligence — Deep Reasoning Revamp

**Lead**: Claude (AI)
**Dev 2**: Senior Dev (via coordination.md)
**Started**: 2026-03-11
**Status**: Planning

---

## 1. The Problem

The current B2B churn intelligence system is a Level 1-2 aggregation engine. It counts reviews, detects snapshot deltas, and generates weekly LLM reports. Every intelligence run re-reasons from scratch — no caching, no pattern reuse, no differential analysis. This is:

- **Expensive**: Full LLM synthesis on every run even when nothing changed
- **Shallow**: No temporal velocity, no archetype matching, no causal inference
- **Commodity**: Any competitor can build aggregation dashboards

The goal is to reach Level 5 (ecosystem-wide pattern recognition) while spending tokens only on novel insights, not re-deriving known patterns.

---

## 2. The Reasoning Depth Hierarchy

| Level | Name | What It Answers | Example |
|-------|------|-----------------|---------|
| L1 | Aggregation | "What happened?" | "Vendor X has 15 negative reviews" |
| L2 | Correlation | "When and why together?" | "Reviews spiked 300% after pricing change on March 1st" |
| L3 | Comparative | "How bad is this vs peers?" | "40% more severe than Vendor Y before their exodus" |
| L4 | Predictive | "What will happen next?" | "Series B + missing SSO + competitor launch = 90-day churn risk" |
| L5 | Ecosystem | "What's shifting structurally?" | "Entire legacy CRM category being displaced, not just vendor switching" |

**Current state**: L1 mature, L2 narrow, L3-L5 absent.

---

## 3. The Stratified Reasoning Engine (Core Architecture)

### 3.1 Three Cognitive Modes

```
┌────────────────────────────────────────────────────────────┐
│                    STRATIFIED REASONER                      │
│                                                            │
│  ┌──────────┐    ┌───────────────┐    ┌─────────────────┐  │
│  │  RECALL  │───>│ RECONSTITUTE  │───>│     REASON      │  │
│  │ (cached) │    │   (patch)     │    │  (full LLM)     │  │
│  │  ~0 tok  │    │  ~medium tok  │    │  ~full tok      │  │
│  └──────────┘    └───────────────┘    └─────────────────┘  │
│       │                 │                     │            │
│       │     confidence  │     diff too        │  no match  │
│       │     > 0.9       │     large or        │  or        │
│       │     + fresh     │     surprise        │  surprise  │
│       ▼                 ▼                     ▼            │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              METACOGNITIVE MONITOR                  │   │
│  │  confidence decay | surprise detection | cost track │   │
│  │  falsification watch | exploration budget (10-15%)  │   │
│  └─────────────────────────────────────────────────────┘   │
└────────────────────────────────────────────────────────────┘
```

**Trigger logic** (when to escalate):

| Mode | Trigger Condition | Token Cost | Use Case |
|------|-------------------|-----------|----------|
| Recall | `semantic_match.confidence > 0.9 AND not stale AND not falsified` | ~0 | Known pattern, fresh data |
| Reconstitute | `episodic_match found AND diff_ratio < 0.3` | ~30% of full | Same pattern + small delta |
| Reason | `no match OR diff_ratio >= 0.3 OR surprise_trigger OR exploration_sample` | ~100% | Novel pattern, store result |

### 3.2 Dual Memory System

#### Semantic Memory (Postgres — fast, compressed)

**Purpose**: Generalized patterns. "Pricing shocks cause churn in mid-market SaaS."

```sql
CREATE TABLE reasoning_semantic_cache (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    pattern_sig     TEXT NOT NULL,           -- e.g. "pricing_shock_v2"
    pattern_class   TEXT NOT NULL,           -- archetype class
    conclusion      JSONB NOT NULL,          -- structured conclusion
    confidence      FLOAT NOT NULL,          -- 0.0-1.0
    reasoning_steps JSONB NOT NULL,          -- [{step, weight, evidence_ids}]
    boundary_conditions JSONB NOT NULL,      -- {valid_for_size, valid_for_category, ...}
    falsification_conditions JSONB,          -- ["competitor_price_drop", "positive_support_trend"]
    uncertainty_sources TEXT[],              -- ["small_sample", "missing_competitor_data"]
    decay_half_life_days INT DEFAULT 90,
    conclusion_type TEXT,                    -- for distribution tracking
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    last_validated_at TIMESTAMPTZ DEFAULT NOW(),
    validation_count INT DEFAULT 1,
    invalidated_at  TIMESTAMPTZ,            -- NULL = active
    UNIQUE(pattern_sig)
);

CREATE INDEX idx_rsc_pattern ON reasoning_semantic_cache(pattern_sig) WHERE invalidated_at IS NULL;
CREATE INDEX idx_rsc_class ON reasoning_semantic_cache(pattern_class) WHERE invalidated_at IS NULL;
```

**Confidence decay formula**:
```
effective_confidence = confidence * 2^(-(days_since_validated / decay_half_life_days))
```

#### Episodic Memory (Neo4j — rich, specific)

**Purpose**: Full reasoning traces with context. Queried when semantic memory is low confidence.

```
Node Types:
  (:ReasoningTrace {id, vendor, category, created_at, conclusion_type, confidence})
  (:EvidenceNode {id, type, source, value, review_id?, event_id?, timestamp})
  (:ConclusionNode {id, claim, confidence, evidence_chain})
  (:FalsificationCondition {id, condition, monitoring_query})

Relationships:
  (:ReasoningTrace)-[:SUPPORTED_BY]->(:EvidenceNode)
  (:ReasoningTrace)-[:CONCLUDED]->(:ConclusionNode)
  (:ReasoningTrace)-[:FALSIFIED_BY]->(:FalsificationCondition)
  (:EvidenceNode)-[:CONFIRMS|CONTRADICTS|NOVEL]->(:EvidenceNode)
```

**Embedding index**: Each ReasoningTrace gets a vector embedding of its evidence signature for similarity search.

### 3.3 Differential Reasoning Engine

When new data arrives for a vendor/category already in cache:

```
1. Retrieve best semantic match (by pattern_sig or embedding similarity)
2. Load episodic trace (full evidence graph)
3. Classify each evidence node against new data:
   - CONFIRMED: New data supports old evidence (skip)
   - CONTRADICTED: New data conflicts (must reason)
   - MISSING: Old evidence no longer present (must reason)
   - NOVEL: New evidence not in original graph (must reason)
4. If (contradicted + novel) / total < 0.3:
   → Reconstitute: LLM patches the conclusion with only the delta
   → Update semantic cache confidence
   → Append to episodic trace
5. If ratio >= 0.3:
   → Full Reason: Complete LLM analysis
   → Store new semantic + episodic entries
   → Mark old entries as superseded
```

### 3.4 Metacognitive Monitor

Tracks the health of the reasoning system itself:

```sql
CREATE TABLE reasoning_metacognition (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    period_start    TIMESTAMPTZ NOT NULL,
    period_end      TIMESTAMPTZ NOT NULL,
    total_queries   INT DEFAULT 0,
    recall_hits     INT DEFAULT 0,          -- semantic cache served
    reconstitute_hits INT DEFAULT 0,        -- patched from episodic
    full_reasons    INT DEFAULT 0,          -- expensive LLM calls
    surprise_escalations INT DEFAULT 0,     -- forced by anomaly
    exploration_samples INT DEFAULT 0,      -- random deep-reasoning
    total_tokens_saved INT DEFAULT 0,       -- estimated vs full-reason-every-time
    total_tokens_spent INT DEFAULT 0,
    conclusion_type_distribution JSONB,     -- {"pricing_shock": 0.4, "feature_gap": 0.25, ...}
    cache_quality_score FLOAT,              -- % of cache hits that matched full-reason result
    created_at      TIMESTAMPTZ DEFAULT NOW()
);
```

**Surprise detection** (the "boredom" algorithm):
- Maintain rolling distribution of conclusion types
- If new conclusion would be in bottom 5% of distribution → escalate to full Reason
- If rare combination of normally-common types → escalate
- Budget: 10-15% of queries get random full-reasoning regardless (drift detection)

**Falsification watcher**:
- Each cached conclusion stores "what would prove this wrong"
- Nightly job checks falsification conditions against new data
- If triggered → invalidate cache entry → force re-reasoning on next query

### 3.5 Hierarchical Abstraction Tiers

```
┌─────────────────────────────────────────────────────────┐
│  TIER 4: MARKET DYNAMICS         (re-reason quarterly)  │
│  "AI-native tools displacing legacy CRM"                │
│  Cached as: market_context tags for categories          │
├─────────────────────────────────────────────────────────┤
│  TIER 3: CATEGORY PATTERNS       (re-reason monthly)    │
│  "PM tools: pricing #1 driver, 3 active archetypes"     │
│  Cached as: category baselines + archetype distribution │
├─────────────────────────────────────────────────────────┤
│  TIER 2: VENDOR ARCHETYPES       (re-reason weekly)     │
│  "Vendor X matches 'pricing shock' at 87% confidence"   │
│  Cached as: vendor archetype match + evidence graph     │
├─────────────────────────────────────────────────────────┤
│  TIER 1: VENDOR STATE            (cache/daily refresh)  │
│  "Vendor X: churn density 25%, urgency 7.2, 3 new"     │
│  Cached as: deterministic metrics (no LLM needed)       │
└─────────────────────────────────────────────────────────┘
```

**Inheritance rule**: Lower tiers receive higher-tier conclusions as priors (context tags), never re-derive them. When Tier 2 analyzes Vendor X (legacy CRM), it retrieves Tier 4's `ai_disruption_pressure: high` tag as input, not re-reasoning about market dynamics.

**Re-reasoning triggers by tier**:
- T4: Calendar-based (quarterly) OR market structure anomaly
- T3: Calendar-based (monthly) OR category-level surprise
- T2: Weekly intelligence run OR vendor-level surprise OR falsification trigger
- T1: Daily snapshot refresh (deterministic, no LLM)

---

## 4. Workstream Breakdown

### WS0: Stratified Reasoning Engine (FOUNDATION)

**Must complete before all others. Everything routes through this.**

| Sub | Component | Dependencies | Deliverables |
|-----|-----------|-------------|-------------|
| 0A | StratifiedReasoner class | None | Core dispatch: recall/reconstitute/reason |
| 0B | Semantic memory (Postgres) | 0A | Table + query/upsert/decay methods |
| 0C | Episodic memory (Neo4j) | 0A | Node types + embedding index + similarity search |
| 0D | Differential engine | 0B, 0C | Evidence diff classifier + LLM patch prompt |
| 0E | Metacognitive monitor | 0A, 0B | Distribution tracker + surprise detection + cost log |
| 0F | Falsification watcher | 0B | Nightly job checking invalidation conditions |
| 0G | Hierarchical tier config | 0A | Tier definitions + inheritance rules + re-reason triggers |

**Key files to create**:
```
atlas_brain/reasoning/
  stratified_reasoner.py    -- Core 3-mode dispatch (0A)
  semantic_cache.py         -- Postgres semantic memory (0B)
  episodic_store.py         -- Neo4j episodic memory (0C)
  differential.py           -- Evidence diff + patch engine (0D)
  metacognition.py          -- Monitor + surprise + cost (0E)
  falsification.py          -- Nightly falsification checker (0F)
  tiers.py                  -- Hierarchical tier config (0G)
```

**Key files to modify**:
```
atlas_brain/reasoning/graph.py          -- Wire stratified reasoner into reasoning pipeline
atlas_brain/autonomous/tasks/b2b_churn_intelligence.py -- Route through stratified reasoner
atlas_brain/reasoning/context_aggregator.py -- Fill _fetch_b2b_churn() stub
```

---

### WS1: Temporal Reasoning Engine (L2-L3)

**First real content flowing through the stratified reasoner.**

| Sub | Component | Depends On | Deliverables |
|-----|-----------|-----------|-------------|
| 1A | Velocity/acceleration | WS0A | Compute rate-of-change from b2b_vendor_snapshots |
| 1B | Rolling percentiles | WS0B | 25th/50th/75th by category (materialized view) |
| 1C | Z-score anomaly | 1A, 1B | Statistical significance testing on spikes |
| 1D | Recency weighting | WS0A | Exponential decay on review age within window |
| 1E | Lag analysis | 1A | Cross-metric lead/lag detection (does DM churn precede mass churn?) |
| 1F | Seasonal correction | 1B | Q4 budget cycle, renewal patterns |

**Produces**: Temporal reasoning traces cached in both memories. "Vendor X urgency accelerating at 0.3/day, z-score 2.4 (p < 0.02), matches pre-churn velocity signature."

**How it connects to WS0**: Each temporal analysis generates a reasoning graph. On repeat analysis, the diff engine checks: did velocity change? Did z-score cross a threshold? If not → recall cached conclusion.

---

### WS2: Churn Archetypes (L3)

**First reusable cross-vendor patterns. This is where caching pays off.**

| Sub | Component | Depends On | Deliverables |
|-----|-----------|-----------|-------------|
| 2A | Archetype definitions | WS1 | Define 6-8 archetypes with signature profiles |
| 2B | Pattern matching | WS0D, 2A | Score vendors against archetype signatures |
| 2C | Historical library | WS0C, 2B | Build episodic archive of past archetype matches |
| 2D | Cross-vendor baselines | WS1B | Category-normalized "how bad is this?" |
| 2E | Temporal pattern lib | WS1A, 2C | "This matches pre-churn signature of 12 vendors" |

**Initial archetype taxonomy**:
```
1. pricing_shock       -- Sudden price increase → complaint spike → competitor mentions
2. feature_gap         -- Competitor launches key feature → "missing X" reviews surge
3. acquisition_decay   -- Post-acquisition quality decline → support complaints → churn
4. leadership_redesign -- New VP Product → UI overhaul → "new interface" complaints
5. integration_break   -- API change breaks workflows → "integration" pain spike
6. support_collapse    -- Support quality drop → response time complaints → trust erosion
7. category_disruption -- New entrant class (e.g., AI-native) → incumbents lose narrative
8. compliance_gap      -- Regulatory requirement unmet → enterprise segments leave
```

**How it connects to WS0**: Archetypes become the primary `pattern_sig` in semantic cache. Once "pricing_shock" is characterized for CRM mid-market, any new CRM vendor showing similar signals → recall, not reason.

---

### WS3: B2B Knowledge Graph (L4 foundation)

**Entity-relationship model enabling multi-hop reasoning.**

| Sub | Component | Depends On | Deliverables |
|-----|-----------|-----------|-------------|
| 3A | Entity model | WS0C | Node types: Vendor, Company, Product, Person, Event |
| 3B | Relation types | 3A | USES, COMPETES_WITH, SWITCHED_TO, INTEGRATES_WITH, etc. |
| 3C | SQL → Graph sync | 3A, 3B | Ingest from existing b2b_* tables into Neo4j |
| 3D | Multi-hop queries | 3C | Cypher queries: Company→Vendor→Competitor→Feature chains |
| 3E | Integration graph | 3C | Build tool-to-tool dependency map from review text |

**How it connects to WS0**: Graph provides evidence nodes for episodic memory. Multi-hop queries become reasoning steps in the trace. "Company Y → uses Vendor X → Vendor X losing to Vendor W → Vendor W launched SSO → Company Y at risk" stored as a 4-hop reasoning chain.

---

### WS4: Trigger Event Detection (L4)

**External signal detection that feeds causal reasoning.**

| Sub | Component | Depends On | Deliverables |
|-----|-----------|-----------|-------------|
| 4A | Event taxonomy | WS2A | Define trigger types: funding, leadership, compliance, launch, pricing |
| 4B | Event ingestion | 4A | Sources: Crunchbase, LinkedIn, press, Wayback Machine |
| 4C | Event-signal correlation | 4B, WS1A | Match trigger events to review spike timestamps |
| 4D | Composite triggers | WS3D, 4C | Multi-condition rules: "funding + missing_feature + competitor_launch" |
| 4E | Risk scoring | 4D, WS2B | Probability model: trigger combo → churn likelihood |

**How it connects to WS0**: Trigger events become falsification conditions. If cached conclusion says "Vendor X stable" but funding event detected at competitor → falsification watcher invalidates cache → forces re-reasoning.

---

### WS5: Ecosystem Pattern Recognition (L5)

**Category-level and market-level intelligence.**

| Sub | Component | Depends On | Deliverables |
|-----|-----------|-----------|-------------|
| 5A | Category health metrics | WS1B, WS2D | HHI concentration, category churn velocity, growth rate |
| 5B | Semantic drift detection | WS0C | Review language shift analysis (embeddings over time) |
| 5C | Market structure classifier | 5A | Consolidating / fragmenting / displacing / stable |
| 5D | News → B2B bridge | WS4B | Wire news_intelligence signals into churn context |
| 5E | "State of Category" reports | 5A-5D | Quarterly ecosystem intelligence output |

**How it connects to WS0**: Ecosystem conclusions cached at Tier 4 (quarterly re-reason). All lower tiers inherit market context as priors. "AI displacement pressure: high" flows down to every CRM vendor analysis without re-derivation.

---

### WS6: Inference Engine & Narrative Layer

**Ties everything together. The brain that generates sellable intelligence.**

| Sub | Component | Depends On | Deliverables |
|-----|-----------|-----------|-------------|
| 6A | Wire B2B into reasoning agent | WS0A, WS3D | Fill _fetch_b2b_churn() stub in context_aggregator |
| 6B | Rule engine | WS4D | Hard rules for threshold triggers + auto-escalation |
| 6C | LLM narrative synthesis | WS0D, WS2B | "Vendor X matches pricing_shock archetype; 87% match to 12 historical cases" |
| 6D | Evidence chains | WS0C | Every claim linked to source review/event/metric |
| 6E | Intervention auto-trigger | WS4E, 6B | Signal threshold → intervention pipeline (existing 3-stage) |
| 6F | Explainability layer | 6D | Audit trail: "why was this vendor ranked #1?" |

---

## 5. Phase Dependencies (Critical Path)

```
Phase 1 ──► Phase 2 ──► Phase 3 ──► Phase 4 ──► Phase 5
  WS0        WS0          WS1         WS3         WS5
  (A,B,C)    (D,E,F,G)    + WS2       + WS4       + WS6

Timeline:
  P1: Core engine + both memories
  P2: Diff engine + metacognition + tiers
  P3: First real intelligence (temporal + archetypes)
  P4: Deep reasoning (graph + triggers + causal rules)
  P5: Full stack (ecosystem + narrative + intervention)
```

### Phase 1 → Phase 2 dependency:
- P1 delivers the 3-mode dispatch + storage
- P2 adds the intelligence to decide WHICH mode (diff engine needs both memories working)

### Phase 2 → Phase 3 dependency:
- P2 delivers differential reasoning + hierarchical tiers
- P3 is the first content that flows through the engine (temporal metrics + archetypes)
- P3 populates the cache; without P2, there's nothing to cache against

### Phase 3 → Phase 4 dependency:
- P3 delivers archetype signatures + temporal baselines
- P4 needs archetypes to classify trigger event relevance ("funding event for a 'pricing_shock' vendor is different from a 'feature_gap' vendor")
- P4 needs temporal baselines to detect "abnormal" vs "seasonal"

### Phase 4 → Phase 5 dependency:
- P4 delivers causal chains + trigger events
- P5 aggregates across vendors/categories (needs the per-vendor causal reasoning to be cached, not re-derived)
- P5's narrative layer needs evidence chains from P4

---

## 6. What Already Exists (Reusable)

| Component | Current Location | Reuse in |
|-----------|-----------------|----------|
| Daily vendor snapshots | `b2b_vendor_snapshots` table | WS1 (temporal source data) |
| Displacement edges | `b2b_displacement_edges` table | WS3 (graph edges), WS2 (archetype evidence) |
| Change events | `b2b_change_events` table | WS1 (velocity input), WS4 (trigger correlation) |
| Keyword signals | `b2b_keyword_signals` table | WS4 (external signal source) |
| Company signals | `b2b_company_signals` table | WS3 (graph edges: Company→Vendor) |
| Graphiti/Neo4j | `memory/rag_client.py` | WS0C (episodic memory), WS3 (entity graph) |
| Reasoning agent | `reasoning/graph.py` | WS6A (wire B2B context in) |
| Intervention pipeline | `services/intervention_pipeline.py` | WS6E (auto-trigger) |
| News intelligence | `autonomous/tasks/news_intake.py` | WS5D (bridge to B2B) |
| Vendor registry | `services/vendor_registry.py` | WS3 (entity normalization) |
| Churn pressure score | `b2b_churn_intelligence.py` | WS1 (baseline for temporal comparison) |
| Skill prompts | `skills/digest/b2b_*.md` | WS6C (narrative templates) |
| LLM usage tracking | `services/tracing.py` | WS0E (metacognitive cost tracking) |

---

## 7. Cost Model

### Current state (no caching):
- Weekly intelligence run: ~50K-100K tokens per vendor (full LLM synthesis)
- 60 vendors × 100K tokens = 6M tokens/week
- At Claude Sonnet rates: ~$18/week, ~$72/month

### With stratified reasoning (target):
- Week 1 (cold cache): Same as current (~6M tokens)
- Week 2+: 70-80% cache hits → ~1.2-1.8M tokens/week
- Surprise escalations (10-15%): ~0.6-0.9M tokens
- Total steady-state: ~2-3M tokens/week (~$6-9/week)
- **Savings: 50-70% ongoing**

### Token budget allocation:
- 60% Recall (semantic cache) — 0 tokens
- 20% Reconstitute (differential patch) — ~30% of full cost per query
- 10% Full Reason (novel patterns) — full cost
- 10% Exploration budget (random deep-reason on "solved" cases)

---

## 8. Risk Register

| Risk | Impact | Mitigation |
|------|--------|-----------|
| Cache staleness → wrong conclusions | High | Confidence decay + falsification watchers + exploration budget |
| Over-caching → miss novel patterns | High | Surprise detection + 10-15% exploration budget |
| Neo4j becomes bottleneck | Medium | Semantic cache (Postgres) handles 60%+ of queries; Neo4j only for episodic |
| Archetype taxonomy too rigid | Medium | LLM-assisted archetype discovery in WS2; taxonomy evolves |
| Diff engine misclassifies evidence | Medium | Threshold tuning; log diff classifications for human review |
| Phase 1 takes too long → blocks everything | High | Minimal viable reasoner first (recall + reason only); add reconstitute in P2 |

---

## 9. Success Metrics

| Phase | Metric | Target |
|-------|--------|--------|
| P1 | Stratified reasoner deployed, both memories operational | Functional |
| P2 | Cache hit rate on repeat vendor analyses | > 60% |
| P3 | Token savings vs full-reason-every-time | > 40% |
| P3 | Archetype match accuracy (human-validated sample) | > 80% |
| P4 | Trigger events detected before review spike | > 50% of cases |
| P5 | Category-level reports generated with < 20% token cost of per-vendor sum | Verified |
| All | Conclusion quality: cache-hit vs full-reason delta | < 5% divergence |

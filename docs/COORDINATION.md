# Dev Coordination — Stratified Reasoning Build

**Lead**: Claude (AI) — architecture, core engine, integration
**Dev 2**: Senior Dev — investigations, parallel workstreams, data layer
**Plan**: `docs/STRATIFIED_REASONING_ARCHITECTURE.md`

---

## How This Works

- Claude writes tasks here with context. Dev 2 picks them up.
- Dev 2 logs findings/blockers here. Claude reads on next session.
- Status: `OPEN` → `IN_PROGRESS` → `DONE` / `BLOCKED`
- Priority: `P0` (blocking), `P1` (needed this phase), `P2` (nice to have)

---

## Current Phase: PRE-BUILD INVESTIGATIONS

Before Phase 1 starts, we need answers to these questions. They inform the engine design.

---

### TASK-001: Audit Neo4j/Graphiti Schema Capacity [P0] [DONE]
**Assigned**: Dev 2
**Context**: We plan to use the existing Neo4j instance (currently serving conversation memory via Graphiti) for episodic reasoning traces AND the B2B entity graph. Need to understand:

1. What's the current Neo4j version and resource allocation? (`docker inspect atlas-neo4j`)
2. How much data is in there now? (node count, relationship count, DB size on disk)
3. Does Graphiti use a specific schema/label convention that we need to avoid colliding with?
4. Can we create a separate Neo4j database within the same instance for B2B entities, or do we share the default DB?
5. Is the Neo4j APOC plugin installed? (needed for batch imports and graph algorithms)
6. What's the vector index situation? Does Graphiti already have one? Can we add a second for reasoning trace embeddings?

**Where to look**:
- `docker-compose.yml` — Neo4j container config
- `atlas_brain/memory/rag_client.py` — How Graphiti talks to Neo4j (HTTP wrapper, not direct Bolt)
- Graphiti wrapper: `docker exec atlas-graphiti-wrapper` — check its config

**Output**: Post findings below.

**Findings**:
```
AUDIT DATE: 2026-03-11

=== 1. NEO4J VERSION & RESOURCE ALLOCATION ===

Container: atlas-neo4j
Image: neo4j:5.26-community (5.26.21)
Edition: Community
Status: running, healthy (since 2026-03-03)
Created: 2026-02-16
Compose project: atlas-memory (from /home/juan-canfield/Desktop/atlas-memory/docker-compose.graphiti.yml)
Network: atlas-memory_graphiti-net (bridge, IP 172.24.0.3)
Ports: 7474 (HTTP browser), 7687 (Bolt)
Auth: neo4j/password123
Restart policy: unless-stopped

Memory allocation (all defaults -- no custom config):
  - Heap: dynamic (auto-calculated, not pinned)
  - Page cache: 512 MB
  - Transaction total max: 10.83 GiB
  - Off-heap transaction max: 2 GB
  - Current usage: 818 MiB / 61.91 GiB host RAM (0.18% CPU)

Volumes:
  - atlas-memory_neo4j_data -> /data (522 MB on disk)
  - atlas-memory_neo4j_logs -> /logs

Plugins installed:
  - APOC 5.26.21 (apoc.jar in /var/lib/neo4j/plugins/)

=== 2. CURRENT DATA VOLUME ===

Total nodes:         1,637
Total relationships: 2,120

Nodes by label:
  Episodic: 1,425
  Entity:     212
  Community:    0
  Saga:         0

Relationships by type:
  MENTIONS:   1,877
  RELATES_TO:   243

Nodes by group_id:
  atlas-conversations / Episodic: 1,403
  atlas-conversations / Entity:     177
  atlas-business / Entity:           25
  atlas-business / Episodic:         19
  test / Entity:                      5
  atlas-test / Entity:                5
  test / Episodic:                    2
  atlas-test / Episodic:              1

Disk usage: 522 MB

=== 3. GRAPHITI SCHEMA / LABEL CONVENTIONS ===

Graphiti uses these node labels (MUST NOT collide):
  - Entity    (properties: name, created_at, uuid, group_id, labels, summary, name_embedding)
  - Episodic  (properties: source_description, source, content, name, valid_at, created_at, uuid, group_id, entity_edges)
  - Community (properties: name, group_id, uuid)
  - Saga      (properties: name, group_id, uuid)

Graphiti uses these relationship types:
  - RELATES_TO  (properties: expired_at, invalid_at, target_node_uuid, source_node_uuid, fact, valid_at, episodes, created_at, fact_embedding, uuid, group_id, name)
  - MENTIONS    (properties: group_id, uuid)
  - HAS_EPISODE (properties: group_id, uuid)
  - HAS_MEMBER  (properties: uuid)
  - NEXT_EPISODE (properties: group_id, uuid)

Graphiti fulltext indexes:
  - episode_content (Episodic: content, source, source_description, group_id)
  - node_name_and_summary (Entity: name, summary, group_id)
  - community_name (Community: name, group_id)
  - edge_name_and_fact (RELATES_TO: name, fact, group_id)

Graphiti RANGE indexes: 27 total (on uuid, group_id, created_at, name, expired_at, valid_at, invalid_at for various labels/rels)

SAFE LABELS for B2B: Any label NOT in {Entity, Episodic, Community, Saga} is safe.
Recommended: B2BEntity, B2BRelation, ReasoningTrace, Archetype, etc.
SAFE RELATIONSHIP TYPES: Anything NOT in {RELATES_TO, MENTIONS, HAS_EPISODE, HAS_MEMBER, NEXT_EPISODE}.

=== 4. MULTI-DATABASE SUPPORT ===

Databases present:
  - neo4j (default, online)
  - system (online)

IMPORTANT: Neo4j Community Edition does NOT support creating additional databases.
Only the default "neo4j" database is available. Enterprise Edition is required
for multi-database (CREATE DATABASE).

IMPLICATION: B2B entities must share the default "neo4j" database with Graphiti.
Collision avoidance strategy: use distinct labels (e.g., B2BVendor, B2BSignal)
and/or a distinct group_id prefix (e.g., "b2b-" or "reasoning-"). Graphiti
already namespaces everything via group_id, so new data should do the same.

=== 5. APOC PLUGIN ===

APOC is installed and working.
  Version: 5.26.21 (matches Neo4j version)
  Location: /var/lib/neo4j/plugins/apoc.jar

Available for: batch imports (apoc.periodic.iterate), graph algorithms,
path expansion, JSON import/export, data generation.

=== 6. VECTOR INDEX SITUATION ===

Current vector indexes: NONE (0 VECTOR type indexes)

However, embeddings ARE stored on nodes and edges:
  - Entity nodes: name_embedding property (1024 dimensions, all 212 entities have it)
  - RELATES_TO edges: fact_embedding property (1024 dimensions)
  - Embedding model: mixedbread-ai/mxbai-embed-large-v1 (1024-dim, sentence-transformers on CPU)

Graphiti currently does vector search via cosine_similarity in application code
(fetches embeddings and computes in Python), NOT via Neo4j native vector index.

CAN WE ADD VECTOR INDEXES? Yes. Neo4j 5.x supports native vector indexes via:
  CREATE VECTOR INDEX entity_embedding FOR (n:Entity) ON (n.name_embedding)
  OPTIONS {indexConfig: {`vector.dimensions`: 1024, `vector.similarity_function`: 'cosine'}}

This would be beneficial for:
  a) Faster vector search (native vs application-side)
  b) A second vector index for reasoning trace embeddings on new labels

No conflicts -- adding vector indexes on new labels (e.g., ReasoningTrace) is
completely independent of Graphiti's existing fulltext/range indexes.

=== 7. GRAPHITI WRAPPER CONFIG ===

Graphiti wrapper runs LOCALLY (not in Docker) at port 8001:
  cd atlas-memory/graphiti-wrapper && ./venv/bin/uvicorn main:app --host 0.0.0.0 --port 8001

.env config:
  LLM: qwen3:32b via Ollama (OPENAI_BASE_URL=http://localhost:11434/v1)
  Neo4j: bolt://localhost:7687 (neo4j/password123)
  Embedder: mixedbread-ai/mxbai-embed-large-v1 (1024-dim, CPU)
  Reranker: BAAI/bge-reranker-v2-m3 (cross-encoder, CPU)

Atlas RAG client (rag_client.py) connects via HTTP to the wrapper:
  - POST /search (vector search with reranking)
  - POST /messages (conversation episode ingestion)
  - GET /entities/{name}/edges (graph traversal)
  - POST /traverse (multi-hop traversal)
  - GET /healthcheck

=== SUMMARY & RECOMMENDATIONS ===

1. The instance is healthy, lightly loaded (1,637 nodes, 522 MB disk, 818 MiB RAM).
   Plenty of capacity for B2B entities + reasoning traces.

2. APOC is installed -- ready for batch imports and graph algorithms.

3. No vector indexes exist yet. Should create them for performance. Both for
   existing Graphiti data AND for new reasoning trace embeddings.

4. Cannot create a separate database (Community Edition limitation). Must share
   the "neo4j" database. Use distinct labels + group_id prefixes to avoid collision.

5. Existing embedding dimension is 1024 (mxbai-embed-large-v1). New data should
   use the same model/dimension for consistency, or use a different property name
   if a different embedding model is needed.

6. Graphiti's group_id namespacing is clean. Current groups: atlas-conversations,
   atlas-business, test, atlas-test. New groups like "b2b-intelligence" or
   "reasoning-traces" would be completely isolated.
```

---

### TASK-002: Profile Existing Snapshot Data Volume [P0] [DONE]
**Assigned**: Dev 2
**Context**: The temporal reasoning engine (WS1) depends on `b2b_vendor_snapshots` having enough history. Need to understand data density.

Run these queries against the DB (port 5433, atlas/atlas):
```sql
-- How many snapshots total?
SELECT COUNT(*) FROM b2b_vendor_snapshots;

-- Date range?
SELECT MIN(snapshot_date), MAX(snapshot_date) FROM b2b_vendor_snapshots;

-- Snapshots per vendor (top 20)?
SELECT vendor_name, COUNT(*) as days
FROM b2b_vendor_snapshots
GROUP BY vendor_name
ORDER BY days DESC
LIMIT 20;

-- Average columns populated per snapshot?
SELECT
  COUNT(*) as total,
  COUNT(pressure_score) as has_pressure,
  COUNT(dm_churn_rate) as has_dm_churn,
  COUNT(churn_density) as has_density,
  COUNT(avg_urgency) as has_urgency
FROM b2b_vendor_snapshots;

-- Change events: how many and what types?
SELECT event_type, COUNT(*) FROM b2b_change_events GROUP BY event_type ORDER BY COUNT(*) DESC;

-- Displacement edges over time?
SELECT computed_date, COUNT(*) FROM b2b_displacement_edges GROUP BY computed_date ORDER BY computed_date;
```

**Output**: Post query results below.

**Findings**:
```
Date: 2026-03-11

=== CORE SNAPSHOT QUERIES ===

Q1: Total snapshots: 54

Q2: Date range: 2026-03-09 to 2026-03-09 (single day only)

Q3: Snapshots per vendor (top 20) -- ALL vendors have exactly 1 day:
  Intercom, Slack, Copper, Zoom, Teamwork, Rippling, Monday.com,
  Insightly, Amazon Web Services, Palo Alto Networks, Azure, HappyFox,
  Workday, CrowdStrike, Microsoft Teams, Basecamp, Mailchimp, Looker,
  Nutshell, Smartsheet -- all 1 day each. 54 vendors total.

Q4: Column population (adapted -- pressure_score/dm_churn_rate don't exist in schema):
  total=54, has_density=54, has_urgency=54, has_positive_pct=54,
  has_recommend=54, has_top_pain=53, has_top_competitor=42
  --> Very good fill rate. Only top_competitor has ~22% NULLs.

Q5: Change events: 0 rows (b2b_change_events table is EMPTY)

Q6: Displacement edges over time:
  2026-03-08: 50
  2026-03-09: 57
  2026-03-10: 50
  2026-03-11: 52
  --> 4 days of displacement data, ~50-57 edges/day, 209 total

=== BONUS: ENRICHED REVIEWS ===

Q7: Enriched reviews: 4,940

Q8: Distinct vendors with enriched reviews: 55

Q9: Enriched reviews per vendor (top 20):
  Notion=276, Shopify=220, Asana=189, Mailchimp=184, Salesforce=180,
  Slack=162, Amazon Web Services=162, Jira=159, Zoom=152, Azure=130,
  ClickUp=128, HubSpot=128, Trello=128, Smartsheet=115, Zendesk=113,
  Gusto=107, Tableau=104, WooCommerce=104, Wrike=103, Monday.com=103

=== ASSESSMENT ===

CRITICAL ISSUE: Only 1 day of snapshot history (2026-03-09).
The temporal reasoning engine needs multi-day history to detect trends.
Snapshot pipeline needs to run daily to accumulate data.

POSITIVE: Enriched review corpus is substantial (4,940 reviews across
55 vendors). Displacement edges have 4 days of data (209 edges).
Column fill rate in snapshots is excellent (98-100% for most fields).

Schema note: The COORDINATION.md queries reference columns
`pressure_score` and `dm_churn_rate` that do not exist in the
b2b_vendor_snapshots table. Actual columns include: churn_density,
avg_urgency, positive_review_pct, recommend_ratio, top_pain,
top_competitor, pain_count, competitor_count, displacement_edge_count,
high_intent_company_count.
```

---

### TASK-003: Benchmark LLM Cost for Current Intelligence Run [P1] [DONE]
**Assigned**: Dev 2
**Context**: We need a baseline to measure token savings against. The stratified reasoner's value prop is "spend 50-70% fewer tokens." Need to know current spend.

1. Check `llm_usage` table for recent b2b_churn_intelligence runs:
```sql
SELECT span_name, model_name,
       SUM(input_tokens) as inp, SUM(output_tokens) as out,
       SUM(cost_usd) as cost, COUNT(*) as calls
FROM llm_usage
WHERE span_name LIKE '%churn_intelligence%'
   OR span_name LIKE '%b2b%intelligence%'
GROUP BY span_name, model_name
ORDER BY cost DESC;
```

2. Check how many vendors get full LLM synthesis per run:
```sql
SELECT COUNT(DISTINCT vendor_name) FROM b2b_churn_signals
WHERE last_computed_at > NOW() - INTERVAL '7 days';
```

3. Check total token spend across all B2B tasks in last 30 days:
```sql
SELECT span_name, SUM(input_tokens + output_tokens) as total_tokens, SUM(cost_usd) as cost
FROM llm_usage
WHERE span_name LIKE '%b2b%'
GROUP BY span_name
ORDER BY total_tokens DESC;
```

**Output**: Post numbers below.

**Findings**:
```
Completed 2026-03-11.

--- Query 1: llm_usage for b2b intelligence runs ---
         span_name          |                 model_name                 |  inp  | out  |   cost   | calls
----------------------------+--------------------------------------------+-------+------+----------+-------
 b2b.churn_intelligence.run | stelterlab/Qwen3-30B-A3B-Instruct-2507-AWQ | 81829 | 2176 | 0.000000 |     6
 b2b.churn_intelligence.run |                                            | 21979 |  920 | 0.000000 |     2
(2 rows)

Notes: 8 total calls logged. Model is a 30B MoE (A3B active params) quantized.
~103K input + ~3K output tokens per full run across all calls. Cost is $0 (local vLLM).

--- Query 2: Vendors with full LLM synthesis in last 7 days ---
 count
-------
    55
(1 row)

Notes: 55 distinct vendors receiving full LLM synthesis per run.

--- Query 3: Total B2B token spend (all time) ---
         span_name          | total_tokens |   cost
----------------------------+--------------+----------
 b2b.churn_intelligence.run |       106904 | 0.000000
(1 row)

Notes: Only one B2B span exists in llm_usage. ~107K tokens total, all local ($0).

--- Query 4: llm_usage table schema ---
Columns: id (uuid PK), span_name (varchar 256), operation_type (varchar 64, default 'llm_call'),
model_name (varchar 256), model_provider (varchar 64), input_tokens (int), output_tokens (int),
total_tokens (int), cost_usd (numeric 10,6), duration_ms (int), ttft_ms (int),
inference_time_ms (int), tokens_per_second (real), status (varchar 32), metadata (jsonb),
created_at (timestamptz).
Indexes: pkey on id, idx on created_at DESC, idx on model_provider+created_at, idx on span_name+created_at.

--- Query 5: Enrichment/triage/extraction token spend ---
(0 rows)

Notes: No spans matching '%enrich%', '%triage%', or '%extraction%' exist.
These operations are either not instrumented or use different span names.

--- Query 6: Total spend last 30 days (all tasks) ---
 total_cost | total_tokens
------------+--------------
   0.617661 |       239018
(1 row)

Notes: $0.62 total across ALL tasks in 30 days. 239K tokens. The non-zero cost
comes from non-B2B tasks (likely cloud LLM calls for email/reasoning). B2B is $0 (local).

=== BASELINE SUMMARY FOR STRATIFIED REASONER ===
- B2B intelligence: ~107K tokens/full-run, 55 vendors, 8 LLM calls, $0 (local vLLM)
- Per-vendor average: ~107K/55 = ~1,945 tokens per vendor synthesis
- System-wide 30-day: 239K tokens, $0.62 (cloud portion only)
- Token savings target (50-70%): would save ~54-75K tokens per B2B run
- Cost savings: negligible for B2B (already $0 local), but latency/throughput gains matter
- Enrichment pipeline NOT instrumented in llm_usage -- spans missing
```

---

### TASK-004: Evaluate Embedding Model for Reasoning Traces [P1] [DONE]
**Assigned**: Dev 2
**Context**: Episodic memory needs embedding-based similarity search. We already have `all-MiniLM-L6-v2` (384-dim) loaded for the intent router. Questions:

1. Is MiniLM sufficient for reasoning trace similarity, or do we need a larger model?
2. What's the current VRAM usage? Can we fit a second embedding model if needed?
3. Test: embed 5 sample reasoning descriptions and check cosine similarity. Do similar patterns cluster?

Sample descriptions to embed:
- "Vendor X pricing shock: 30% price increase, 45% complaint rate, competitor Y launched cheaper alternative"
- "Vendor Z pricing shock: 25% price increase, 38% complaint rate, competitor W offers free tier"
- "Vendor A feature gap: missing SSO, enterprise customers leaving for Vendor B"
- "Vendor C support collapse: response time doubled, CSAT dropped 40%"
- "Vendor D acquisition: acquired by BigCorp, product team departed, quality declining"

If descriptions 1 and 2 have cosine_sim > 0.8, MiniLM is probably fine. If < 0.7, we need a bigger model.

**Output**: Similarity matrix below.

**Findings**:
```
COMPLETED BY DEV 2, 2026-03-11

=== TWO MODELS TESTED ===

--- Model A: all-MiniLM-L6-v2 (384-dim, CPU) ---
  Same-archetype (pricing_1 vs pricing_2): 0.7763
  Cross-archetype avg:                     0.3929
  Separation gap:                          0.3834
  VERDICT: MARGINAL -- below 0.8 threshold

--- Model B: mxbai-embed-large-v1 (1024-dim, CPU) ---
  Same-archetype (pricing_1 vs pricing_2): 0.8282
  Cross-archetype avg:                     0.5845
  Separation gap:                          0.2437
  VERDICT: PASSES -- same-archetype > 0.8

Full mxbai similarity matrix:
  pricing_1 vs pricing_2:       0.8282 (SAME archetype)
  pricing_1 vs feature_gap:     0.5332 (DIFF)
  pricing_1 vs support_collapse: 0.5553 (DIFF)
  pricing_1 vs acquisition:     0.5609 (DIFF)
  feature_gap vs support_col:   0.6651 (DIFF)
  support_col vs acquisition:   0.6080 (DIFF)

=== DECISION ===

Use mxbai-embed-large-v1 (1024-dim) for reasoning trace embeddings.

Rationale:
1. Clears the 0.8 same-archetype threshold (MiniLM does not)
2. Already loaded in graphiti-wrapper process (CPU, zero extra VRAM)
3. Matches existing Neo4j embedding dimension (1024-dim) -- no dim mismatch
4. Atlas already has SentenceTransformerEmbedding service that can load it
5. Graphiti-wrapper exposes it via HTTP if needed (POST /search)
6. vLLM cannot be used for embeddings (serving Qwen3-30B text gen only)

VRAM impact: None. mxbai runs on CPU (~500MB RAM). GPU is fully allocated
to vLLM (Qwen3-30B) + ASR (Nemotron).

Recommended similarity threshold for cache recall: 0.75
  (below 0.8282 same-archetype, above 0.6651 cross-archetype max)
```

---

### TASK-005: Map Current Intelligence Skill Prompts [P1] [DONE]
**Assigned**: Dev 2
**Context**: WS6 needs to upgrade the skill prompts for narrative generation. Before we modify them, need a complete map.

List all skill files in `atlas_brain/skills/digest/` that relate to B2B intelligence. For each:
- File name
- What it generates (report type)
- Input data it expects
- Whether it has any "compare to historical" or "match archetype" instructions (probably not, but check)
- Line count (to estimate modification effort)

**Output**: Table below.

**Findings**:
```
COMPLETE MAP OF B2B-RELATED SKILL PROMPT FILES
Date: 2026-03-11
===============================================

## 1. Core B2B Intelligence Skills (atlas_brain/skills/digest/)

### b2b_churn_extraction.md (259 lines)
- Generates: Structured churn signal JSON from a single B2B software review
- Input: Single review JSON (vendor_name, product_name, rating, review_text, pros, cons,
  reviewer_title, company_size_raw, reviewer_industry, source_name, source_weight)
- Output fields: churn_signals, urgency_score (0-10), reviewer_context, pain_category,
  competitors_mentioned, contract_context, budget_signals, use_case, sentiment_trajectory,
  buyer_authority, timeline
- Historical/archetype instructions: NONE. Pure per-review extraction, no comparison logic.

### b2b_churn_intelligence.md (231 lines)
- Generates: Weekly B2B churn intelligence synthesis report (JSON)
- Input: 18 data sets including vendor_churn_scores, high_intent_companies,
  competitive_displacement, pain_distribution, feature_gaps, budget_signals,
  sentiment_trajectory, buyer_authority, churning_companies, prior_reports, known_companies
- Output fields: executive_summary, weekly_churn_feed[], vendor_scorecards[],
  displacement_map[], category_insights[], timeline_hot_list[]
- Historical/archetype instructions: YES -- has "trend" field using prior_reports
  (>5pp churn change or >1.0 urgency change = worsening/improving). Uses sentiment_trajectory
  for direction. But NO archetype matching -- purely comparison to own prior data.

### b2b_churn_triage.md (52 lines)
- Generates: Quick boolean triage (signal: true/false, confidence 1-10, reason)
- Input: Review JSON (vendor_name, source, rating, summary, review_text, pros, cons)
- Output: {signal: bool, confidence: int, reason: str}
- Historical/archetype instructions: NONE. Stateless gate for extraction pipeline.

### b2b_exploratory_overview.md (66 lines)
- Generates: Supplemental JSON with exploratory_summary + timeline_hot_list
- Input: Same trimmed payload as churn intelligence (high_intent_companies,
  timeline_signals, competitive_displacement, quotable_evidence, known_companies)
- Output: {exploratory_summary: str (120 words max), timeline_hot_list: [...]}
- Historical/archetype instructions: NONE. No trend or archetype logic.

## 2. Blog Content Generation Skills (atlas_brain/skills/digest/)

### b2b_blog_post_generation.md (325 lines) -- LARGEST FILE
- Generates: Full SEO-optimized blog post with metadata (title, seo_title,
  seo_description, target_keyword, secondary_keywords, faq[], content)
- Input: Blueprint JSON with topic_type (10 types: vendor_alternative, vendor_showdown,
  churn_report, migration_guide, vendor_deep_dive, market_landscape, pricing_reality_check,
  switching_story, pain_point_roundup, best_fit_guide), sections, available_charts,
  quotable_phrases, data_context, related_posts
- Historical/archetype instructions: NONE. No trend comparison or archetype matching.
  Extensive SEO, AEO (Answer Engine Optimization), featured snippet, and linking rules.

## 3. Campaign/Outreach Generation Skills (atlas_brain/skills/digest/)

### b2b_campaign_generation.md (134 lines)
- Generates: ABM outreach content (email_cold, linkedin, email_followup)
- Input: Company-level churn data (churning_from, pain_categories, competitors_considering,
  urgency, seat_count, contract_end, role_type, reviewer_title, industry, company_size,
  key_quotes, feature_gaps, integration_stack, sentiment_direction, selling context, channel)
- Historical/archetype instructions: NONE.

### b2b_campaign_sequence.md (95 lines)
- Generates: Next email in a multi-step ABM sequence
- Input: Template vars: company_name, company_context, selling_context, current_step,
  max_steps, days_since_last, engagement_summary, previous_emails
- Historical/archetype instructions: NONE. Engagement-signal-driven (opened/clicked/replied).

### b2b_challenger_outreach.md (122 lines)
- Generates: Outreach to challenger companies GAINING market share, selling them
  qualified intent leads
- Input: challenger_name, contact_name, contact_role, signal_summary (total_leads,
  by_buying_stage, role_distribution, pain_driving_switch, incumbents_losing,
  seat_count_signals, feature_mentions), key_quotes, tier, selling context, channel
- Historical/archetype instructions: NONE.

### b2b_challenger_sequence.md (105 lines)
- Generates: Next email in challenger intel campaign sequence
- Input: Template vars (company_name, company_context, selling_context,
  engagement_summary, previous_emails)
- Historical/archetype instructions: NONE. Per-step intelligence layering
  (Step 2: incumbents, Step 3: pain+seats, Step 4: break-up).

### b2b_vendor_outreach.md (124 lines)
- Generates: Outreach to vendors LOSING customers, selling them churn intelligence
- Input: vendor_name, contact_name, contact_role, signal_summary (total_signals,
  high_urgency_count, pain_distribution, competitor_distribution, feature_gaps,
  timeline_signals, trend_vs_last_month), key_quotes, tier, selling context, channel
- Historical/archetype instructions: NONE. Uses trend_vs_last_month for urgency
  calibration but no archetype matching.

### b2b_vendor_sequence.md (105 lines)
- Generates: Next email in vendor retention campaign sequence
- Input: Template vars (company_name, company_context, selling_context,
  engagement_summary, previous_emails)
- Historical/archetype instructions: NONE. Per-step intelligence layering
  (Step 2: competitor displacement, Step 3: feature gaps+trend, Step 4: break-up).

### b2b_onboarding_sequence.md (82 lines)
- Generates: Next email in onboarding sequence for new Atlas Intel customers
- Input: Template vars (company_name, company_context, selling_context,
  engagement_summary)
- Historical/archetype instructions: NONE. Step-based (welcome, insights,
  feature highlight, trial wrap-up).

## 4. B2B Skills Outside digest/ (atlas_brain/skills/b2b/)

### b2b/product_profile_synthesis.md (53 lines)
- Generates: Product profile knowledge card (summary + pain_addressed scores)
- Input: vendor_name, product_category, total_reviews, avg_rating, strengths[],
  weaknesses[], use_cases[], integrations[], competitive_data, pain_categories[]
- Output: {summary: str, pain_addressed: {category: 0.0-1.0, ...}}
- Historical/archetype instructions: NONE. Stateless synthesis.

## 5. Adjacent Intelligence Skills (in digest/, not B2B-prefixed but related)

### competitive_intelligence.md (161 lines)
- Generates: Consumer product complaint vulnerability analysis
- Input: brand_health[], competitive_flows[], feature_gaps[], buyer_personas[],
  sentiment_landscape[], safety_signals[], loyalty_churn[], prior_reports[]
- Historical/archetype instructions: YES -- trend detection via prior_reports.
  No archetype matching.

### subcategory_intelligence.md (122 lines)
- Generates: Amazon subcategory intel for 3 audiences (seller, dropshipper, new brand)
- Historical/archetype instructions: NONE.

### amazon_seller_campaign_generation.md (137 lines)
- Generates: Outreach to Amazon sellers with category intelligence
- Historical/archetype instructions: NONE.

### amazon_seller_campaign_sequence.md (108 lines)
- Generates: Next email in Amazon seller outreach sequence
- Historical/archetype instructions: NONE.

## 6. Skills in atlas_brain/skills/intelligence/ -- NOT B2B related
- report.md, autonomous_narrative_architect.md, prompt_to_report.md,
  report_builder.md, adaptive_intervention.md, simulated_evolution.md
- None contain B2B/churn/vendor/displacement content.

## 7. context_aggregator.py -- _fetch_b2b_churn() (lines 306-359)

_fetch_b2b_churn() is FULLY IMPLEMENTED (not a stub):
- entity_type="company": uses entity_id directly as company hint
- entity_type="contact": resolves UUID -> email -> domain -> company hint
  (skips free email domains)
- Queries b2b_churn_signals WHERE company appears in company_churn_list
  (JSONB ILIKE match), returns top 5 by avg_urgency_score
- Returns: vendor_name, product_category, avg_urgency_score,
  top_pain_categories, top_competitors, decision_maker_churn_rate,
  price_complaint_rate

## 8. reasoning/graph.py -- Reasoning Agent Structure (491 lines)

Manual state machine with 8 nodes:
1. _node_triage: event priority classification (Haiku LLM)
2. _node_aggregate_context: calls context_aggregator.aggregate_context()
3. _node_check_lock: entity lock coordination
4. _node_reason: deep reasoning (Claude Sonnet)
5. _node_plan_actions: safe action filtering (5 allowed tools)
6. _node_execute_actions: dispatches planned actions
7. _node_synthesize: human-readable summary (Haiku)
8. _node_notify: ntfy push notification

** IMPORTANT GAP FOUND **: b2b_churn data is fetched by aggregate_context()
(line 41: "b2b_churn": []) but _node_aggregate_context (lines 191-199)
does NOT copy it into reasoning state. The state update maps crm_context,
email_history, voice_turns, calendar_events, sms_messages, graph_facts,
recent_events, market_context, news_context -- but OMITS b2b_churn.
The reasoning agent currently ignores B2B churn data even when available.

## SUMMARY TABLE

| File                              | Lines | Historical? | Archetype? |
|-----------------------------------|------:|:-----------:|:----------:|
| b2b_churn_triage.md               |    52 |     No      |     No     |
| b2b_churn_extraction.md           |   259 |     No      |     No     |
| b2b_churn_intelligence.md         |   231 |    YES      |     No     |
| b2b_exploratory_overview.md       |    66 |     No      |     No     |
| b2b_blog_post_generation.md       |   325 |     No      |     No     |
| b2b_campaign_generation.md        |   134 |     No      |     No     |
| b2b_campaign_sequence.md          |    95 |     No      |     No     |
| b2b_challenger_outreach.md        |   122 |     No      |     No     |
| b2b_challenger_sequence.md        |   105 |     No      |     No     |
| b2b_vendor_outreach.md            |   124 |     No      |     No     |
| b2b_vendor_sequence.md            |   105 |     No      |     No     |
| b2b_onboarding_sequence.md        |    82 |     No      |     No     |
| b2b/product_profile_synthesis.md  |    53 |     No      |     No     |
| competitive_intelligence.md       |   161 |    YES      |     No     |
| subcategory_intelligence.md       |   122 |     No      |     No     |
| amazon_seller_campaign_gen.md     |   137 |     No      |     No     |
| amazon_seller_campaign_seq.md     |   108 |     No      |     No     |
| TOTAL                             | 2,281 |             |            |

KEY FINDINGS FOR WS6:
1. Only 2/17 prompts use prior_reports for historical comparison
   (b2b_churn_intelligence, competitive_intelligence)
2. ZERO prompts have archetype matching -- greenfield for WS6
3. b2b_churn_intelligence.md (231 lines) is the primary WS6 upgrade target
4. b2b_blog_post_generation.md (325 lines) may need archetype-driven angles
5. reasoning/graph.py omits b2b_churn from state (gap at lines 191-199)
```

---

## Phase 1: Core Engine + Both Memories (WS0A + WS0B + WS0C)

**Goal**: Build the stratified reasoner dispatch, semantic cache (Postgres), and episodic store (Neo4j). After Phase 1, the system can Recall or Reason (Reconstitute comes in Phase 2 with the differential engine).

**Investigation-informed decisions**:
- Neo4j Community Edition — no multi-database. Use distinct labels (`ReasoningTrace`, `EvidenceNode`, `ConclusionNode`) + `group_id: "b2b-reasoning"` for namespace isolation (TASK-001)
- APOC available for batch ops. No vector indexes yet — we create one for trace embeddings (TASK-001)
- **mxbai-embed-large-v1 (1024-dim)** for reasoning trace embeddings — clears 0.8 same-archetype threshold (0.83), matches existing Neo4j embedding dimension, already loaded in graphiti-wrapper on CPU (TASK-004, Dev 2 finding)
- Similarity threshold for cache recall: **0.75** (below 0.83 same-archetype, above 0.67 cross-archetype max)
- Only 1 day of snapshot history — WS1 temporal analysis is data-gated. WS0 can proceed without it (TASK-002)
- B2B intelligence costs $0 (local vLLM). Savings value = latency + throughput, not dollars (TASK-003)
- `_fetch_b2b_churn()` in context_aggregator.py is implemented but graph.py never copies it into reasoning state — fix in WS6A later (TASK-005)
- 0/17 skill prompts have archetype matching — greenfield for WS2+WS6 (TASK-005)

---

### P1-001: Database Migration — reasoning tables [P0] [DONE]
**Assigned**: Dev 2
**Estimate**: Small (1 file, ~50 lines)

Create `atlas_brain/storage/migrations/123_reasoning_semantic_cache.sql`:

```sql
CREATE TABLE IF NOT EXISTS reasoning_semantic_cache (
    id                      UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    pattern_sig             TEXT NOT NULL,
    pattern_class           TEXT NOT NULL,
    vendor_name             TEXT,
    product_category        TEXT,
    conclusion              JSONB NOT NULL,
    confidence              FLOAT NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
    reasoning_steps         JSONB NOT NULL DEFAULT '[]',
    boundary_conditions     JSONB NOT NULL DEFAULT '{}',
    falsification_conditions JSONB DEFAULT '[]',
    uncertainty_sources     TEXT[] DEFAULT '{}',
    decay_half_life_days    INT DEFAULT 90,
    conclusion_type         TEXT,
    evidence_hash           TEXT,
    created_at              TIMESTAMPTZ DEFAULT NOW(),
    last_validated_at       TIMESTAMPTZ DEFAULT NOW(),
    validation_count        INT DEFAULT 1,
    invalidated_at          TIMESTAMPTZ,
    UNIQUE(pattern_sig)
);

CREATE INDEX idx_rsc_pattern ON reasoning_semantic_cache(pattern_sig) WHERE invalidated_at IS NULL;
CREATE INDEX idx_rsc_class ON reasoning_semantic_cache(pattern_class) WHERE invalidated_at IS NULL;
CREATE INDEX idx_rsc_vendor ON reasoning_semantic_cache(vendor_name) WHERE invalidated_at IS NULL;
CREATE INDEX idx_rsc_confidence ON reasoning_semantic_cache(confidence DESC) WHERE invalidated_at IS NULL;

CREATE TABLE IF NOT EXISTS reasoning_metacognition (
    id                          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    period_start                TIMESTAMPTZ NOT NULL UNIQUE,
    period_end                  TIMESTAMPTZ NOT NULL,
    total_queries               INT DEFAULT 0,
    recall_hits                 INT DEFAULT 0,
    reconstitute_hits           INT DEFAULT 0,
    full_reasons                INT DEFAULT 0,
    surprise_escalations        INT DEFAULT 0,
    exploration_samples         INT DEFAULT 0,
    total_tokens_saved          INT DEFAULT 0,
    total_tokens_spent          INT DEFAULT 0,
    conclusion_type_distribution JSONB DEFAULT '{}',
    cache_quality_score         FLOAT,
    created_at                  TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_rm_period ON reasoning_metacognition(period_start DESC);
```

**Verification**: `psql -p 5433 -U atlas -d atlas -f atlas_brain/storage/migrations/123_reasoning_semantic_cache.sql`

**Findings**:
```
DONE by Dev 2, 2026-03-11.

Migration 123 was already taken (consumer_brand_registry). Used 130 instead.
File: atlas_brain/storage/migrations/130_reasoning_semantic_cache.sql

Tables created:
- reasoning_semantic_cache: 17 columns, 4 partial indexes, UNIQUE(pattern_sig),
  CHECK(confidence 0-1). Verified with \d.
- reasoning_metacognition: 13 columns, 1 index on period_start DESC. Verified.

Run command: psql -h localhost -p 5433 -U atlas -d atlas -f atlas_brain/storage/migrations/130_reasoning_semantic_cache.sql
Result: CREATE TABLE x2, CREATE INDEX x5. Clean.
```

---

### P1-002: Build semantic_cache.py [P0] [DONE]
**Assigned**: Claude
**Estimate**: Medium (~200 lines)
**Depends on**: P1-001

Create `atlas_brain/reasoning/semantic_cache.py`. Core class: `SemanticCache`.

**Interface**:
```python
class SemanticCache:
    def __init__(self, pool: DatabasePool):
        ...

    async def lookup(self, pattern_sig: str) -> CacheEntry | None:
        """Recall mode. Fetch active entry, apply confidence decay formula.
        Return None if effective_confidence < 0.5 (stale)."""

    async def store(self, entry: CacheEntry) -> None:
        """Store new conclusion. INSERT ON CONFLICT(pattern_sig) DO UPDATE."""

    async def validate(self, pattern_sig: str, new_confidence: float | None = None) -> None:
        """Bump last_validated_at + validation_count. Optionally update confidence."""

    async def invalidate(self, pattern_sig: str, reason: str) -> None:
        """Soft-delete: set invalidated_at. Log reason."""

    async def lookup_by_class(self, pattern_class: str, vendor: str | None = None) -> list[CacheEntry]:
        """Find all active entries for a pattern class, optionally filtered by vendor."""

    async def get_cache_stats(self) -> dict:
        """Return counts for metacognition tracking."""
```

**Key behaviors**:
- `lookup()` applies: `effective_confidence = confidence * 2^(-(days_since_validated / decay_half_life_days))`
- All queries filter `WHERE invalidated_at IS NULL`
- `evidence_hash` = SHA256 of sorted evidence IDs — used by diff engine (Phase 2) to detect changed evidence

**Pattern**: Use `self._pool.fetchrow()` / `self._pool.execute()` — same as all existing DB code.

---

### P1-003: Build episodic_store.py [P0] [DONE]
**Assigned**: Claude
**Estimate**: Medium-Large (~250 lines)
**Depends on**: P1-001

Create `atlas_brain/reasoning/episodic_store.py`. Core class: `EpisodicStore`.

**Neo4j connection**: Direct Bolt driver (`neo4j` package, port 7687, auth `neo4j/password123`). Preferred over Graphiti HTTP wrapper — more control for vector index creation and custom queries.

**Interface**:
```python
class EpisodicStore:
    def __init__(self, bolt_url="bolt://localhost:7687", auth=("neo4j", "password123")):
        ...

    async def store_trace(self, trace: ReasoningTrace) -> str:
        """Store full reasoning trace (nodes + rels). Return trace_id."""

    async def find_similar(self, embedding: list[float], threshold: float = 0.75, limit: int = 5) -> list[ReasoningTrace]:
        """Vector similarity search on trace embeddings using native Neo4j vector index."""

    async def get_trace(self, trace_id: str) -> ReasoningTrace | None:
        """Load full trace with evidence nodes and conclusions."""

    async def get_traces_for_vendor(self, vendor_name: str, limit: int = 10) -> list[ReasoningTrace]:
        """All traces for a vendor, newest first."""

    async def ensure_indexes(self) -> None:
        """Create vector index + range indexes if not exists. Call on app startup."""
```

**Node labels** (distinct from Graphiti's Entity/Episodic/Community/Saga):
- `ReasoningTrace` — `{id, vendor_name, category, created_at, conclusion_type, confidence, pattern_sig, trace_embedding, group_id}`
- `EvidenceNode` — `{id, type, source, value, review_id, event_id, timestamp, group_id}`
- `ConclusionNode` — `{id, claim, confidence, evidence_chain, group_id}`

**Relationships**: `SUPPORTED_BY`, `CONCLUDED`, `SUPERSEDES`

**Vector index** (in `ensure_indexes()`):
```cypher
CREATE VECTOR INDEX reasoning_trace_embedding IF NOT EXISTS
FOR (n:ReasoningTrace) ON (n.trace_embedding)
OPTIONS {indexConfig: {`vector.dimensions`: 1024, `vector.similarity_function`: 'cosine'}}
```

**Embedding model**: mxbai-embed-large-v1 (1024-dim). Access via:
- Option A: Import `SentenceTransformerEmbedding` and load mxbai locally (CPU, ~500MB RAM)
- Option B: HTTP call to graphiti-wrapper's embedding endpoint if exposed
- Decision: Option A preferred (no network hop, already proven in graphiti-wrapper)

**group_id**: All reasoning nodes use `"b2b-reasoning"`.

---

### P1-004: Build stratified_reasoner.py [P0] [DONE]
**Assigned**: Claude
**Estimate**: Medium (~180 lines)
**Depends on**: P1-002, P1-003

Create `atlas_brain/reasoning/stratified_reasoner.py`. Core class: `StratifiedReasoner`.

Phase 1 minimal version — **Recall** and **Reason** modes only. Reconstitute (differential patching) added in Phase 2.

**Interface**:
```python
class StratifiedReasoner:
    def __init__(self, cache: SemanticCache, episodic: EpisodicStore):
        ...

    async def analyze(self, vendor_name: str, evidence: dict, force_reason: bool = False) -> ReasoningResult:
        """
        Main entry. Decides recall or reason:
        1. Build pattern_sig from vendor + evidence_hash
        2. Semantic cache lookup (recall)
           - Hit + confidence > threshold → return cached
        3. Episodic similarity search (find related traces as LLM context)
        4. Full LLM reason → store in both memories → return
        """

    async def _recall(self, pattern_sig: str) -> ReasoningResult | None:
    async def _reason(self, vendor_name: str, evidence: dict, prior_traces: list) -> ReasoningResult:
    def _build_pattern_sig(self, vendor_name: str, evidence: dict) -> str:
    def _compute_evidence_hash(self, evidence: dict) -> str:
```

**ReasoningResult**:
```python
@dataclass
class ReasoningResult:
    mode: str          # "recall" | "reconstitute" | "reason"
    conclusion: dict
    confidence: float
    pattern_sig: str
    evidence_hash: str
    tokens_used: int
    cached: bool
    trace_id: str | None = None
```

**LLM call**: Use `call_llm_with_skill()` with inline system prompt for Phase 1. Full skill file added in WS6.

**Metacognition**: After each `analyze()`, increment counters in `reasoning_metacognition` (recall_hits / full_reasons / tokens_spent / tokens_saved).

---

### P1-005: Integration wiring — app startup [P1] [DONE]
**Assigned**: Claude
**Estimate**: Small (~30 lines)
**Depends on**: P1-004

1. **`atlas_brain/main.py`** (startup): Init `SemanticCache`, `EpisodicStore`, `StratifiedReasoner` as app-scoped singletons. Call `episodic.ensure_indexes()`.
2. **`atlas_brain/reasoning/__init__.py`**: Export `get_stratified_reasoner()` factory.
3. **Do NOT wire into `b2b_churn_intelligence.py` yet** — that's Phase 2+ when diff engine is ready.

---

### P1-006: Snapshot pipeline daily accumulation [P0] [DONE]
**Assigned**: Dev 2
**Estimate**: Small (verification + potential scheduler fix)

**Critical gap from TASK-002**: Only 1 day of `b2b_vendor_snapshots`. WS1 needs 14+ days.

1. Verify snapshot pipeline runs daily. Check `scheduled_tasks` for b2b_churn_intelligence schedule.
2. If running but not creating snapshots, check INSERT logic in `b2b_churn_intelligence.py`.
3. If not scheduled daily, fix the cron expression.
4. Confirm `b2b_change_events` has 0 rows because only 1 snapshot day exists (not a code bug). Change detection needs 2+ days of snapshots to compute deltas.

**Verification**: After 2+ days: `SELECT COUNT(DISTINCT snapshot_date) FROM b2b_vendor_snapshots;` should be > 1.

**Findings**:
```
DONE by Dev 2, 2026-03-11.

Pipeline is healthy. No code fix needed.

1. SCHEDULING: b2b_churn_intelligence runs daily at 9 PM (cron: 0 21 * * *).
   Defined in _pipelines.py line 241. Config override: b2b_churn.intelligence_cron.

2. SNAPSHOT INSERT: _persist_vendor_snapshots() in b2b_churn_intelligence.py
   (line 1186). Uses ON CONFLICT (vendor_name, snapshot_date) DO UPDATE (idempotent).
   Controlled by config flag: b2b_churn.snapshot_enabled (default True).

3. CHANGE EVENTS: _detect_change_events() (line 1300) queries for prior
   snapshot WHERE snapshot_date < today. If no prior exists, skips vendor
   (line 1325: if not prior: continue). This confirms 0 change events is
   EXPECTED with 1 day of data -- not a bug.

4. RETENTION: 365-day retention policy (line 1288). Snapshots auto-deleted
   past retention window.

ROOT CAUSE: Pipeline recently started or DB was reset. After tonight's
9 PM run, there will be 2 days. Change events will start generating
on Day 2. No action required.
```

---

### P1-007: End-to-end smoke test [P1] [DONE]
**Assigned**: Claude

File: `tests/test_stratified_reasoner.py` (10 tests)

```
tests/test_stratified_reasoner.py::TestPureLogic::test_evidence_hash_deterministic PASSED
tests/test_stratified_reasoner.py::TestPureLogic::test_evidence_hash_changes PASSED
tests/test_stratified_reasoner.py::TestPureLogic::test_decay_fresh PASSED
tests/test_stratified_reasoner.py::TestPureLogic::test_decay_half_life PASSED
tests/test_stratified_reasoner.py::TestPureLogic::test_decay_two_half_lives PASSED
tests/test_stratified_reasoner.py::test_semantic_cache_store_and_recall PASSED
tests/test_stratified_reasoner.py::test_semantic_cache_invalidation PASSED
tests/test_stratified_reasoner.py::test_semantic_cache_validation_bump PASSED
tests/test_stratified_reasoner.py::test_episodic_store_roundtrip PASSED
tests/test_stratified_reasoner.py::test_cache_stats PASSED
10 passed in 0.43s
```

---

## Task Assignment Summary

| Task | Assignee | Status | Blocked By |
|------|----------|--------|-----------|
| P1-001 | Dev 2 | DONE | — |
| P1-002 | Claude | DONE | P1-001 |
| P1-003 | Claude | DONE | P1-001 |
| P1-004 | Claude | DONE | P1-002, P1-003 |
| P1-005 | Claude | DONE | P1-004 |
| P1-006 | Dev 2 | DONE | — |
| P1-007 | Claude | DONE | P1-005 |

**Phase 1 COMPLETE.** All 7 tasks done. 10/10 tests passing.

---

## Phase 2: Diff Engine + Metacognition + Falsification + Tiers (WS0D + WS0E + WS0F + WS0G)

**Goal**: Complete the stratified reasoning engine. After Phase 2, the system has all 3 cognitive modes (recall/reconstitute/reason), metacognitive monitoring with surprise detection, nightly falsification checks, and hierarchical tier inheritance.

**Code written by Claude (already done)**:
- `atlas_brain/reasoning/differential.py` (WS0D) — Evidence diff classifier, Jaccard list comparison, reconstitute LLM prompt. 5% numeric tolerance, 30% diff threshold for reconstitute vs full reason.
- `atlas_brain/reasoning/metacognition.py` (WS0E) — In-memory state + DB flush, surprise detection (bottom 5% of distribution), 12% exploration budget, rolling distribution cache.
- `atlas_brain/reasoning/falsification.py` (WS0F) — Nightly checker: loads all active cache entries with falsification conditions, evaluates against fresh vendor signals (snapshots, reviews, change events), invalidates on trigger.
- `atlas_brain/reasoning/tiers.py` (WS0G) — 4-tier hierarchy (T1 daily/T2 weekly/T3 monthly/T4 quarterly), inheritance rules, `gather_tier_context()` loads higher-tier priors for LLM context.
- `stratified_reasoner.py` updated — Reconstitute mode wired in, metacognitive monitor integrated, tier_context passthrough to LLM.
- `__init__.py` updated — MetacognitiveMonitor initialized at startup, flushed at shutdown.
- 21/21 tests passing (11 new Phase 2 tests).

---

### P2-001: Register falsification watcher as autonomous task [P1] [DONE]
**Assigned**: Dev 2

The falsification watcher (`atlas_brain/reasoning/falsification.py`) needs to run nightly. Register it as an autonomous task.

1. Add a handler in `atlas_brain/autonomous/tasks/` (new file or extend existing) that:
   - Imports `FalsificationWatcher` from `atlas_brain.reasoning.falsification`
   - Imports `SemanticCache` and creates it with the DB pool
   - Calls `watcher.run_nightly_check()`
   - Returns summary: `{entries_checked, entries_invalidated, triggered_conditions}`

2. Register in `_DEFAULT_TASKS` in `scheduler.py`:
   - Name: `falsification_check`
   - Cron: `0 4 * * *` (4 AM, after nightly_memory_sync at 3 AM)
   - Timeout: 300s
   - Handler: the new handler

3. Verify: `SELECT * FROM scheduled_tasks WHERE task_type = 'falsification_check';`

**Files to modify**:
- `atlas_brain/autonomous/tasks/__init__.py` (register handler)
- `atlas_brain/autonomous/scheduler.py` (add to _DEFAULT_TASKS)

**Findings**:
```
DONE by Dev 2, 2026-03-11.

Created: atlas_brain/autonomous/tasks/falsification_check.py (52 lines)
- Imports FalsificationWatcher + SemanticCache
- Gets DB pool, creates cache + watcher, calls run_nightly_check()
- Returns {entries_checked, entries_invalidated, triggered_conditions}
- Uses _skip_synthesis=True (no LLM summary needed)

Modified: atlas_brain/autonomous/tasks/__init__.py
- Added ("falsification_check", "run", "falsification_check") to _BUILTIN_TASKS

Modified: atlas_brain/autonomous/scheduler.py
- Added falsification_check to _DEFAULT_TASKS
- Cron: 0 4 * * * (4 AM), timeout: 300s
- metadata: {"builtin_handler": "falsification_check"}

Syntax verified. All files pass ast.parse().
```

---

### P2-002: Metacognition flush in autonomous runner [P1] [DONE]
**Assigned**: Dev 2

The MetacognitiveMonitor accumulates state in memory and needs periodic flushing to the DB. Two integration points:

1. **After each b2b_churn_intelligence run**: In `atlas_brain/autonomous/tasks/b2b_churn_intelligence.py`, at the end of the intelligence run, call:
   ```python
   from atlas_brain.reasoning import get_stratified_reasoner
   reasoner = get_stratified_reasoner()
   if reasoner and reasoner._meta:
       await reasoner._meta.flush()
   ```

2. **Verify flush works**: After a manual trigger of any reasoning, check:
   ```sql
   SELECT * FROM reasoning_metacognition ORDER BY period_start DESC LIMIT 5;
   ```

**Findings**:
```
DONE by Dev 2, 2026-03-11.

Modified: atlas_brain/autonomous/tasks/b2b_churn_intelligence.py (line ~2277)
- Inserted flush after tracer.end_span(), before return response
- Wrapped in try/except with logger.debug (non-fatal)
- Imports get_stratified_reasoner lazily to avoid circular imports
```

---

### P2-003: Add b2b_change_events.direction column if missing [P1] [DONE]
**Assigned**: Dev 2

The falsification watcher queries `b2b_change_events.direction` for support trend detection. Verify this column exists:

```sql
SELECT column_name FROM information_schema.columns
WHERE table_name = 'b2b_change_events' AND column_name = 'direction';
```

If missing, add it:
```sql
ALTER TABLE b2b_change_events ADD COLUMN IF NOT EXISTS direction TEXT;
```

Also check `b2b_enriched_reviews.overall_sentiment` exists (used for negative review count in falsification).

**Findings**:
```
DONE by Dev 2, 2026-03-11.

Both columns were MISSING. Created migration 131_falsification_columns.sql.

1. b2b_change_events.direction: Added. ALTER TABLE succeeded.

2. overall_sentiment: The table b2b_enriched_reviews does NOT EXIST.
   Actual table is b2b_reviews. Added overall_sentiment column to b2b_reviews.

3. BUG FIX in falsification.py (line 231):
   - Changed table name: b2b_enriched_reviews -> b2b_reviews
   - Changed timestamp column: created_at -> enriched_at
   (b2b_reviews has no created_at column; enriched_at is the correct timestamp)

Migration: atlas_brain/storage/migrations/131_falsification_columns.sql
Both ALTER TABLEs verified via information_schema.
```

---

## Phase 2 Task Summary

| Task | Assignee | Status | What |
|------|----------|--------|------|
| P2-001 | Dev 2 | DONE | Register falsification watcher as autonomous task |
| P2-002 | Dev 2 | DONE | Wire metacognition flush into intelligence runner |
| P2-003 | Dev 2 | DONE | Verify/add direction column + fix table name bug |
| P2-CODE | Claude | DONE | All 4 modules built + reasoner updated + 21 tests |

**Phase 2 COMPLETE.** All tasks done. 21/21 tests passing.

---

## Blockers Log

| Date | Blocker | Owner | Resolution |
|------|---------|-------|-----------|
| | | | |

---

## Decision Log

| Date | Decision | Rationale | Decided By |
|------|----------|-----------|-----------|
| 2026-03-11 | Use existing Neo4j for episodic memory (not new instance) | Reduce infra complexity; Graphiti already manages it | Claude + Juan |
| 2026-03-11 | Postgres for semantic cache (not Redis) | Need ACID, complex queries, already have connection pool | Claude |
| 2026-03-11 | max_prospects_per_company reduced 10→3 | Only need VP Sales/CRO/VP Marketing for challenger intel | Claude + Juan |
| 2026-03-11 | 8 initial archetypes defined | Covers observed patterns in current displacement data | Claude |
| 2026-03-11 | mxbai-embed-large-v1 (1024-dim) for reasoning trace embeddings | Clears 0.8 same-archetype threshold (0.83), matches Neo4j dim, already on CPU | Dev 2 + Claude |
| 2026-03-11 | Similarity threshold 0.75 for cache recall | Below 0.83 same-archetype, above 0.67 cross-archetype max | Dev 2 |
| 2026-03-11 | Direct Bolt driver for Neo4j (not Graphiti HTTP wrapper) | More control for vector indexes, custom queries, no Graphiti schema constraints | Claude |

---

## Session Notes

### 2026-03-11 (Session 1)
- Completed full audit of current reasoning capabilities (L1 mature, L2 narrow, L3-5 absent)
- User provided stratified reasoning architecture with caching/cost optimization strategy
- Created 8-workstream plan with 5-phase implementation order
- Key insight: WS0 (stratified reasoner) must come first — everything routes through it
- 5 pre-build investigations assigned (mix of Claude + Dev 2)
- Architecture plan logged to `docs/STRATIFIED_REASONING_ARCHITECTURE.md`

### 2026-03-11 (Session 2)
- All 5 pre-build investigations COMPLETE
- TASK-001 (Neo4j audit): Community Edition, 1637 nodes, APOC installed, no vector indexes yet, must share DB with Graphiti via label namespacing
- TASK-002 (snapshot data): CRITICAL — only 1 day of history. WS1 data-gated until 14+ days accumulate
- TASK-003 (LLM cost): ~107K tokens/run, $0 local vLLM. Savings = latency/throughput, not dollars
- TASK-004 (embeddings): Dev 2 tested both MiniLM and mxbai. mxbai wins (0.83 same-archetype, 1024-dim matches Neo4j)
- TASK-005 (skill map): 14 B2B skills, 2,481 lines, 0 have archetype matching. graph.py omits b2b_churn from reasoning state
- **Phase 1 tasks posted**: 7 tasks (P1-001 through P1-007). Dev 2 starts P1-001 + P1-006. Claude starts P1-002/003 once migration lands.

### 2026-03-11 (Session 3)
- **Phase 1 COMPLETE**: All 7 tasks done. Dev 2 landed migration (130) + confirmed snapshot pipeline healthy.
- Claude built semantic_cache.py, episodic_store.py, stratified_reasoner.py. All wired into main.py startup/shutdown. 10/10 tests.
- **Phase 2 COMPLETE (code)**: All 4 modules built in single session:
  - `differential.py`: Evidence diff classifier (confirmed/contradicted/missing/novel), 5% numeric tolerance, Jaccard for lists, 30% threshold
  - `metacognition.py`: In-memory state + DB flush, 12% exploration budget, surprise detection (bottom 5%), distribution caching
  - `falsification.py`: Nightly checker evaluates conditions against fresh signals (snapshots, reviews, change events)
  - `tiers.py`: 4-tier hierarchy with inheritance rules + refresh intervals
  - `stratified_reasoner.py`: Updated with Reconstitute mode, metacognitive monitor, tier_context
- 21/21 tests passing (11 new Phase 2 tests for diff engine, tiers, metacognition)
- **3 integration tasks posted for Dev 2**: P2-001 (register falsification task), P2-002 (metacognition flush), P2-003 (verify column schema)

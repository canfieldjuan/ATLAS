# Intelligence Platform Roadmap

## Current-State Assessment (March 2026)

Audit performed against the live codebase. Each phase item rated as **EXISTS**, **PARTIAL**, or **MISSING**.

### Phase 0: Current-State Hardening -- COMPLETE

| Item | Status | What's There | Remaining Gaps |
|------|--------|-------------|----------|
| Source naming | **EXISTS** | `ReviewSource` enum (17 members) in `sources.py`. Classification sets (`VERIFIED_SOURCES`, `SLUG_SOURCES`, `SEARCH_SOURCES`, etc.), `display_name()` helper, `parse_source_allowlist()`. All consumers migrated. | -- |
| Vendor normalization | **EXISTS** | `b2b_vendor_registry` table (migration 095) with canonical_name + aliases array. `resolve_vendor_name_cached()` used across intelligence synthesis. MCP tools: `list_vendors_registry`, `add_vendor_to_registry`, `add_vendor_alias`. | No normalization at ingest time (still applied at synthesis). No fuzzy matching. |
| Provenance fields | **EXISTS** | Migration 096 adds `source_review_count`, `source_distribution` to `b2b_intelligence` and `b2b_churn_signals`. Intelligence reports and churn signals now back-reference source data. | No enrichment model/version tracking on individual reviews (see Phase 1 parser versioning). |
| Source reliability | **EXISTS** | Migration 097 adds `idx_scrape_log_source_health` composite index. `get_source_health` MCP tool + `/source-health` dashboard endpoint with per-source success rate, block rate, yield metrics, trend comparison. | No automated alerting on degradation. |

### Phase 1: Managed Intelligence Substrate -- COMPLETE

| Item | Status | What's There | Remaining Gaps |
|------|--------|-------------|----------|
| Source orchestration | **EXISTS** | Priority-ordered scheduling, per-source semaphores (4 web / 10 API), retry with exponential backoff, cooldown after blocking, fallback proxy rotation, G2 3-tier fallback (Web Unlocker -> Playwright -> residential). | -- |
| Source capability profiles | **EXISTS** | `capabilities.py` module with `SourceCapabilityProfile` model. `get_source_capabilities` MCP tool + dashboard endpoint. Per-source: access pattern, anti-bot classification, proxy requirements, data quality tier. | -- |
| Scrape telemetry | **EXISTS** | `b2b_scrape_log` table: target_id, source, status, reviews_found/inserted, pages_scraped, errors (JSONB), duration_ms, proxy_type, started_at. | Specific proxy IP, CAPTCHA solve time, and block type classification not persisted. |
| Parser versioning | **EXISTS** | Migration 098 adds `parser_version` to `b2b_scrape_log`. Each parser reports its version string. Enables selective re-extraction when parsers improve. | No automatic re-processing trigger. |

### Phase 2: Canonical Intelligence Model -- IN PROGRESS (Sprint 1 complete)

| Item | Status | What's There | Remaining Gaps |
|------|--------|-------------|----------|
| Canonical entities | **PARTIAL** | `b2b_churn_signals` (per-vendor), `b2b_product_profiles` (vendor knowledge cards), `b2b_keyword_signals` (search volume), `b2b_alert_baselines`. **Sprint 1 adds:** `b2b_displacement_edges` (append-only time-series, migration 099) and `b2b_company_signals` (UPSERT per company-vendor, migration 099). | Pain points, use cases, integrations, reviewer personas still in JSONB. |
| Identity resolution | **PARTIAL** | DB-backed `b2b_vendor_registry` with aliases (Phase 0). `_canonicalize_vendor()` / `_canonicalize_competitor()` applied at synthesis. Review-level dedup via SHA-256 dedup_key. | No vendor merge system. No fuzzy matching. No company-level identity resolution. |
| Confidence scoring | **PARTIAL** | `confidence_score NUMERIC(3,2)` column on `b2b_displacement_edges`. Evidence-based 3-signal scoring (`_compute_displacement_confidence`): mention weight (log-scaled), source diversity, verified source proportion. Source distribution and sample review IDs persisted per edge. | Confidence not yet on `b2b_company_signals`, `b2b_churn_signals`, or `b2b_product_profiles`. Not propagated through full aggregation chain. |

**Phase 2 Sprint 1 (displacement edges + company signals) is code-complete but NOT YET TESTED against a live intelligence run.** Migration 099 has not been applied to production DB. The next intelligence run will be the first live validation.

Phase 2 Sprint 2+ scope (deferred):
- Pain point tables (promote from JSONB)
- Integration/use-case tables
- Reviewer persona aggregations
- Company-level identity resolution (fuzzy matching, aliases)
- Full entity graph edges (vendor -> strong_in_use_case, vendor -> weak_on_pain_category)

### Phase 3: Historical Memory

| Item | Status | What's There | Key Gaps |
|------|--------|-------------|----------|
| Snapshots | **PARTIAL** | `b2b_keyword_signals` is a true weekly time-series with rolling averages and spike detection. Intelligence reports are date-stamped. `b2b_displacement_edges` is append-only with `computed_date` (new -- enables displacement trend queries). | `b2b_churn_signals` and `b2b_product_profiles` are upsert-overwritten -- no history. No vendor health snapshot table. |
| Change events | **PARTIAL** | Trend labels (new/worsening/improving/stable) computed per-run by comparing to prior report. Keyword spike detection. High-urgency real-time ntfy alerts. | Trends embedded in report JSONB, not persisted as queryable events. No event log table. No structural change detection (new competitor, new displacement edge). |
| Historical queries | **PARTIAL** | Prior report comparison (1 cycle back). Keyword rolling averages over 4 weeks. `get_displacement_history` MCP tool enables time-series queries for specific vendor pairs (new). | Cannot query "vendor X's churn density 3 months ago." No arbitrary historical range queries for most entities. |

### Phase 4: Action Feedback Loop

| Item | Status | What's There | Key Gaps |
|------|--------|-------------|----------|
| Campaign outcome tracking | **PARTIAL** | Full ESP webhook ingestion (opened, clicked, bounced, complained). Reply detection with intent classification. `campaign_funnel_stats` materialized view. Campaign audit log. | No meeting_booked, deal_stage, closed_won/lost, pipeline_stage tracking. Campaign outcomes don't connect back to originating churn signals. |
| CRM event ingestion | **MISSING** | -- | No pipeline reads CRM events back into intelligence. No external CRM integration (HubSpot, Salesforce, Pipedrive). |
| Score calibration | **MISSING** | Static `opportunity_score` formula exists. No learning. | Campaign outcomes never update scoring. No A/B testing or calibration infrastructure. |

### Phase 5: Thin Delivery Surfaces

| Item | Status | What's There | Key Gaps |
|------|--------|-------------|----------|
| API-first intelligence | **EXISTS** | Full REST API (dashboard, tenant, campaigns). B2B Churn MCP: 25 tools (was 18, +2 displacement edges, +5 vendor registry/source tools). Intelligence MCP: 17 tools. | -- |
| Alternate delivery | **PARTIAL** | Email digest (weekly tenant reports via Resend). CSV export (signals, reviews, high-intent). ntfy push notifications. | No PDF export. No webhook outbound. No CRM sync. No Slack/Teams integration. |
| UI logic leakage | **CLEAN** | Frontend is display-only. All scoring, aggregation, analysis happen server-side. | -- |

### Phase 6: Analyst Controls

| Item | Status | What's There | Key Gaps |
|------|--------|-------------|----------|
| Correction tools | **PARTIAL** | Campaign suppression management (CRUD). Campaign review queue (bulk approve/reject). Scrape target management (CRUD + MCP). Blog post admin (draft/edit/publish). Vendor alias management (MCP: `add_vendor_alias`, `add_vendor_to_registry`). | No vendor merge tool. No source suppression/quality control. No review-level correction/flagging. |
| Correction persistence | **PARTIAL** | `campaign_audit_log` (immutable campaign events). `campaign_suppressions` (manual overrides). | No general-purpose `data_corrections` table. No audit trail for intelligence corrections. |

### Overall Score

| Phase | Score | Summary |
|-------|-------|---------|
| Phase 0 | **90%** | Canonical source enum, vendor registry, provenance fields, source health metrics all delivered. Ingest-time normalization is the remaining gap. |
| Phase 1 | **90%** | Orchestration, telemetry, capability profiles, parser versioning all exist. Minor telemetry gaps (proxy IP, CAPTCHA solve time). |
| Phase 2 | **45%** | Sprint 1 complete: displacement edges + company signals are first-class tables with confidence scoring. Pain points, use cases, personas still in JSONB. Identity resolution still partial. **Not yet tested against live run.** |
| Phase 3 | **30%** | Displacement edges now provide one time-series entity. Keyword signals have rolling averages. Core vendor metrics still overwritten in place. |
| Phase 4 | 15% | Campaign delivery is tracked but the feedback loop back to scoring is completely absent. |
| Phase 5 | **70%** | API-first is solid (25 + 17 = 42 MCP tools). Missing PDF, webhooks, and CRM push. UI is already clean. |
| Phase 6 | **30%** | Campaign-side + vendor alias controls exist. Intelligence-side corrections still missing. |

### What's Strong Today

1. **Source orchestration** -- priority scheduling, retry, cooldown, concurrency, proxy fallback
2. **Scrape telemetry** -- every run logged with status, duration, proxy type, yield, parser version
3. **API-first delivery** -- full REST + 42 MCP tools, no intelligence logic in frontend
4. **Campaign lifecycle** -- ESP webhooks, reply detection, audit log, funnel analytics
5. **Consumer brand normalization** -- `comparisons.py` pipeline is production-grade
6. **Canonical source identity** -- `ReviewSource` enum, display names, classification sets used everywhere
7. **Vendor identity** -- DB-backed registry with aliases, MCP management tools

### Biggest Structural Gaps

1. **No feedback loop** -- campaign outcomes never improve signal scoring
2. **No historical depth** -- core vendor metrics overwritten in place, no snapshots (displacement edges are the exception)
3. **Remaining JSONB entities** -- pain points, use cases, integrations, reviewer personas not yet first-class
4. **No company identity resolution** -- company names are free-text, no alias/merge system
5. **Phase 2 untested** -- Sprint 1 code is complete but migration 099 not applied, no live intelligence run yet

### Testing checkpoint

Phase 2 Sprint 1 is a good stopping point for end-to-end validation before continuing to Sprint 2. Testing plan:

1. Apply migration 099 to the database
2. Verify tables and indexes exist (`\d b2b_displacement_edges`, `\d b2b_company_signals`)
3. Trigger a manual intelligence run and verify:
   - `displacement_edges_persisted > 0` in the return dict
   - `company_signals_persisted > 0` in the return dict
   - `SELECT count(*) FROM b2b_displacement_edges` returns rows
   - `SELECT count(*) FROM b2b_company_signals` returns rows
   - Confidence scores are reasonable (0.0-1.0 range, higher for multi-source edges)
   - Source distribution JSONB is populated per edge
   - Sample review IDs reference real `b2b_reviews` rows
4. Test MCP tools: `list_displacement_edges`, `get_displacement_history`
5. Test dashboard endpoints: `GET /b2b/dashboard/displacement-edges`, `GET /b2b/dashboard/company-signals`
6. Verify existing intelligence report JSONB still includes displacement data (backwards compatible)
7. Verify existing tests still pass (28 validation + 22 keyword = 50 tests passing)

---

## Line Reference Verification (March 2026)

All file references in the implementation plan were audited against the live codebase. Results:

### Verified Exact (33 references)

All storage migrations, delivery/API endpoints (tenant dashboard, public dashboard, MCP server),
tenant report persistence loop, enrichment processing, and hard-coded values at
`config.py:2082-2088`, `config.py:2200-2203`, `b2b_churn_intelligence.py:28-42` confirmed exact.

### Corrected Line References

| Plan Reference | Actual Location | What's There |
|---------------|----------------|--------------|
| `config.py:2234-2236` (source allowlist) | **2239-2241** | `B2BScrapeConfig.source_allowlist` |
| `config.py:2293-2315` (anti-bot domains) | **2294-2321** | captcha + web_unlocker config (extends 6 lines) |
| `b2b_product_profiles.py:24-35` (pain categories) | **28-39** | `_PAIN_CATEGORIES` list |
| `b2b_scrape_intake.py:160-179` (concurrency) | **163-178** | `_API_SOURCES`, `_WEB_CONCURRENCY`, `_API_CONCURRENCY` |
| `b2b_scrape_intake.py:205-207` (parser.scrape) | **213** | `result = await parser.scrape(target, client)` |
| `b2b_scrape_intake.py:282-291` (_log_scrape call) | **292-296** (success) / **217** (failure) | Two call sites |
| `browser.py:150-163` (scrape_page return) | **169-173** | `BrowserScrapeResult(...)` return in `_do_scrape` |
| `browser.py:1-223` (full module) | **1-275** | File is 275 lines, not 223 |
| `b2b_churn_intelligence.py:110-199` (_validate_report) | **118-214** | Function starts 8 lines later, ends 15 lines later |
| `b2b_churn_intelligence.py:775-819` (category overview) | **789-836** | `_build_deterministic_category_overview()` |
| `b2b_churn_intelligence.py:821-920` (report assembly) | **1295-1370+** | **CRITICAL**: plan says "report assembly" but 821-920 has helper/trimmer functions. Actual assembly is 475 lines later. |
| `b2b_product_profiles.py:156-203` (competitive flows) | **176-217** | `_fetch_competitive_flows()` starts 20 lines later |
| `b2b_campaign_generation.py:80-108` (run function) | **92-124** | `run()` starts at 92 |
| `b2b_campaign_generation.py:111-219` (cold email seq) | **127-218** | `_create_sequence_for_cold_email()` starts at 127 |
| `product_matching.py:16-24` (scoring constants) | **14-26** | `_WEIGHT_*` + `_SEVERITY_WEIGHTS` |
| `VendorDetail.tsx:1-176` | **1-161** | File is 161 lines, not 176 |

### Critical Correction

The plan's Phase 3 insertion point `b2b_churn_intelligence.py:821-920` claims this is the
"report assembly area" where snapshot persistence should be added. **This is wrong.**
Lines 821-920 contain trimming helpers and exploratory payload builders.

The actual report assembly -- where all four deterministic report types are constructed,
combined into `report_types` list, and persisted via `INSERT INTO b2b_intelligence` inside
a transaction -- is at **lines 1295-1370+**. Any Phase 3 snapshot persistence must target
this region, not lines 821-920.

### Path Corrections

- `product_matching.py` is at `atlas_brain/services/b2b/product_matching.py`, not `autonomous/tasks/`
- `b2b_scrape_intake.py` is at `atlas_brain/autonomous/tasks/b2b_scrape_intake.py`
- `browser.py` is at `atlas_brain/services/scraping/browser.py`

---

## Goal

Shift Atlas so the defensible asset is:

- proprietary source acquisition
- normalization and enrichment
- longitudinal intelligence history
- action loops and outcome data

And make the UI:

- a clean delivery surface
- replaceable
- not the primary differentiation

## North-star end state

Atlas becomes an intelligence platform with:

- a durable ingestion and anti-bot layer
- canonical source-normalized intelligence records
- historical trend storage
- scoring, displacement, and trigger engines
- multi-surface delivery: UI, API, alerts, CRM, outbound, reports

Best framing:

- data system first
- delivery surfaces second

---

## Phase 0: Current-state hardening

**Objective:** Stabilize the current moat before expanding it.

**Why first:** The repo already has strong signals of moat in scraping, normalization, and intelligence synthesis. That foundation should be made reliable before adding more product layers.

### Existing foundation to build from

- scrape config and anti-bot controls in `config.py:2224-2321`
- stealth browser and challenge handling in `browser.py:1-223`
- B2B intelligence task in `b2b_churn_intelligence.py:1-260`
- product matching / displacement scoring in `product_matching.py:130-270`
- delivery UI pages in b2b

### Needed changes

**Standardize source naming everywhere**

- one canonical source enum
- alias handling for source names
- one parsing helper used by scrape intake, enrichment, blog, and reporting

**Formalize canonical vendor / competitor normalization**

- extend alias maps
- persist alias decisions
- track confidence on canonicalization

**Add explicit provenance on every derived intelligence object**

- source
- source_url
- review_id
- scraped_at
- enriched_at
- extraction_method
- confidence

**Add reliability metrics per source**

- scrape success rate
- block rate
- captcha solve rate
- parse success rate
- usable review yield

### Deliverables

- source normalization helper module
- canonical vendor registry
- provenance fields enforced in aggregation queries
- source health dashboard endpoint

### Acceptance criteria

- no mixed source labels in downstream reports
- every intelligence row can be traced back to raw evidence
- source-level quality can be measured daily

---

## Phase 1: Turn collection into a managed intelligence substrate

**Objective:** Make collection itself a system, not a set of scrapers.

### Needed changes

**Build a source orchestration layer**

- source priority
- schedule windows
- retry policy
- fallback route policy
- cool-down logic after blocking

**Add source capability profiles**

- API source
- HTML scrape
- JS-rendered scrape
- stealth browser required
- captcha likely
- proxy class required
- expected freshness

**Persist operational scrape telemetry**

New tables should track:

- request attempts
- proxy used
- captcha provider used
- result status
- block type
- solve duration
- extraction yield

**Add parser versioning**

- every extraction should store parser version
- enable reprocessing of raw data when parser improves

### Likely file areas

- `b2b_scrape_intake.py`
- `browser.py`
- `config.py:2224-2321`
- scrape API surface in `b2b_scrape.py`

### Deliverables

- scrape telemetry schema
- source capability registry
- parser version persistence
- operational source health API

### Acceptance criteria

- can answer: which sources are worth the money?
- can answer: which proxy/captcha path works best per domain?
- can re-run extraction when logic improves

---

## Phase 2: Build the canonical intelligence model

**Objective:** Move from "scraped reviews" to "proprietary intelligence graph."

### Needed changes

**Define canonical entities**

- vendor
- product
- competitor
- company
- reviewer persona
- pain point
- churn signal
- displacement edge
- use case
- integration
- source

**Add identity resolution layer**

- vendor aliases
- company aliases
- competitor aliases
- product family matching
- source-specific entity cleanup

**Persist derived graph edges**

Example edge types:

- company -> leaving_vendor
- company -> considering_vendor
- vendor -> displaced_by_vendor
- vendor -> strong_in_use_case
- vendor -> weak_on_pain_category
- source -> supports_claim

**Add confidence and evidence packing**

Every derived record should include:

- confidence score
- evidence count
- evidence samples
- first_seen
- last_seen
- freshness decay weight

### Deliverables

- canonical intelligence schema
- entity resolution layer
- materialized displacement edges
- confidence-scored evidence model

### Acceptance criteria

- same vendor is not fragmented across aliases
- displacement is queryable as structured edges, not just LLM output
- intelligence can be reused across UI, API, reports, and campaigns

---

## Phase 3: Build historical memory and time-series moat

**Objective:** Make the moat cumulative.

**Why:** Anyone can approximate current-state snapshots. Historical intelligence is much harder to recreate.

### Needed changes

**Persist weekly and daily snapshots**

- vendor health
- displacement
- pain clusters
- source mix
- urgency changes
- lead emergence

**Track change events**

- new competitor appearing
- spike in churn intent
- drop in sentiment
- increase in switching mentions
- change in use-case distribution

**Add historical comparison queries**

- vendor now vs 30 days ago
- category now vs quarter ago
- displacement trend over time
- source confidence trend

**Build freshness and drift models**

- stale source detection
- vendor narrative drift
- parser drift
- source coverage gaps

### Likely touchpoints

- intelligence task in `b2b_churn_intelligence.py`
- reporting endpoints in `b2b_tenant_dashboard.py`
- reports UI in `B2BReports.tsx`

### Deliverables

- snapshot tables
- trend materializations
- change-event engine
- time-series APIs

### Acceptance criteria

- can tell not just what is true, but what is changing
- historical depth compounds monthly
- customers rely on trend intelligence, not only snapshots

---

## Phase 4: Close the loop into action systems

**Objective:** Make Atlas learn from downstream use and outcomes.

### Needed changes

**Link intelligence to lead and campaign outcomes**

- campaign sent
- response
- meeting booked
- opportunity opened
- vendor switched
- lost/won outcome

**Add CRM/event ingestion**

- enrich Atlas leads with real sales outcomes
- store outcome labels back against signals

**Build action feedback scoring**

- which signals predict booked meetings?
- which displacement patterns lead to actual churn?
- which source combinations are highest quality?

**Use results to tune rankings**

- lead score calibration
- urgency calibration
- source weighting
- displacement weighting

### Existing adjacent areas

- campaigns UI in `B2BCampaigns.tsx`
- lead pipeline pages in `LeadPipeline.tsx`
- product matching in `product_matching.py:130-270`

### Deliverables

- outcome event schema
- score calibration jobs
- action-performance dashboards
- lead-to-outcome attribution

### Acceptance criteria

- Atlas scores improve from feedback
- recommendations are not static heuristics
- value is measurable in revenue or retention outcomes

---

## Phase 5: Make delivery surfaces explicitly thin

**Objective:** Ensure the UI is not where the moat lives.

### Needed changes

**Promote all core intelligence to APIs first**

- every dashboard block must come from reusable endpoints
- no important logic hidden only in the UI

**Add alternate delivery surfaces**

- API
- email digests
- PDF report export
- webhook delivery
- CRM sync
- Slack / Teams notifications

**Keep the UI conventional**

- tables
- filters
- saved views
- standard charts
- drilldowns
- No exotic front-end moat investment.

**Add embedded / white-label capability**

- if customers can consume Atlas outside your app, the moat is clearly the data layer

### UI direction

The current UI is already conventional and product-ready:

- dashboard in `B2BDashboard.tsx:1-190`
- displacement in `CompetitorDisplacement.tsx:1-73`

That is good. Keep it useful, not magical.

### Deliverables

- intelligence API contracts
- export service
- webhook push service
- embedded report format

### Acceptance criteria

- a customer can receive value without logging into the UI
- UI removal would hurt convenience, not destroy the product

---

## Phase 6: Build proprietary analyst controls

**Objective:** Turn operational knowledge into a durable asset.

### Needed changes

**Add internal analyst tools for:**

- alias corrections
- source suppression
- parser overrides
- quote rejection
- vendor merge/split decisions
- confidence overrides

**Persist all analyst corrections**

- correction reason
- who changed it
- affected records
- timestamp

**Feed corrections back into pipeline defaults**

### Why this matters

Human-tuned production judgment is one of the hardest forms of moat to copy.

### Deliverables

- analyst correction tables
- internal admin endpoints
- correction application layer
- audit trail UI

### Acceptance criteria

- manual knowledge compounds over time
- recurring data issues stop recurring
- output quality improves from usage

---

## Phase 7: Reposition product language around the moat

**Objective:** Make the product narrative match the architecture.

### Messaging direction

**Avoid:**

- "AI dashboard"
- "beautiful UI"
- "review insights portal"

**Use:**

- continuously collected competitive intelligence
- displacement intelligence network
- evidence-backed churn signal graph
- historical vendor risk memory
- action-ready intelligence feeds

### Delivery packaging

**Sell:**

- vendor watchlists
- churn-trigger feeds
- displacement maps
- account-level lead signals
- benchmark reports
- outbound-ready lead packages
- analyst-backed intelligence exports

**Not just:**

- dashboards

---

## Recommended implementation order

### Sprint order

1. Phase 0
2. Phase 1
3. Phase 2
4. Phase 3
5. Phase 4
6. Phase 5
7. Phase 6
8. Phase 7

### If resources are limited

Do these first:

1. scrape telemetry
2. canonical entity model
3. historical snapshots
4. outcome feedback loop
5. API-first delivery

Those five do the most to make the statement true.

---

## Concrete repo-level build directions

### Backend priority areas

- `browser.py`
- `config.py:2060-2321`
- `b2b_churn_intelligence.py`
- `product_matching.py`
- `b2b_tenant_dashboard.py:546-598`
- `b2b_scrape.py`

### UI priority areas

Keep building, but in a thin-client direction:

- `B2BDashboard.tsx`
- `CompetitorDisplacement.tsx`
- `B2BCampaigns.tsx`
- `LeadPipeline.tsx`

**Rule:** logic belongs in backend intelligence services. UI only renders and routes actions.

---

## Success metrics

Track these weekly:

### Collection moat

- scrape success rate by domain
- block rate by domain
- captcha solve success rate
- cost per successful usable record
- time to recover blocked source

### Intelligence moat

- percent of rows with canonical vendor resolution
- percent of outputs with evidence provenance
- displacement precision
- quote verification pass rate
- trend coverage depth

### Product moat

- percent of customer value delivered outside dashboard
- number of triggered actions from intelligence
- conversion from signal to lead to meeting
- retention driven by recurring intelligence, not pageviews

---

## Final test for "100% true"

The statement becomes effectively true when:

- the UI can be rebuilt by a competent team in weeks
- the data pipeline cannot be rebuilt without months of source tuning, parser iteration, anti-bot infrastructure, historical accumulation, and feedback data

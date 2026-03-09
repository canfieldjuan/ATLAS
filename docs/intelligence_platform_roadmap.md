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

### Phase 2: Canonical Intelligence Model -- COMPLETE

| Item | Status | What's There | Remaining Gaps |
|------|--------|-------------|----------|
| Canonical entities | **EXISTS** | `b2b_churn_signals` (per-vendor), `b2b_product_profiles` (vendor knowledge cards), `b2b_keyword_signals` (search volume), `b2b_alert_baselines`. **Sprint 1:** `b2b_displacement_edges` (append-only time-series, migration 099), `b2b_company_signals` (UPSERT per company-vendor, migration 099). **Sprint 2:** `b2b_vendor_pain_points` (UPSERT per vendor-category with confidence scoring, migration 100), `b2b_vendor_use_cases` (UPSERT per vendor-module, migration 100), `b2b_vendor_integrations` (UPSERT per vendor-tool, migration 100). **Sprint 3:** `b2b_vendor_buyer_profiles` (UPSERT per vendor-role-stage, migration 101). | Reviewer personas still in JSONB (deferred -- low ROI vs. buyer profiles). |
| Identity resolution | **PARTIAL** | DB-backed `b2b_vendor_registry` with aliases (Phase 0). `_canonicalize_vendor()` / `_canonicalize_competitor()` applied at synthesis. Review-level dedup via SHA-256 dedup_key. | No vendor merge system. No fuzzy matching. No company-level identity resolution. |
| Confidence scoring | **EXISTS** | Evidence-based 3-signal scoring (`_compute_evidence_confidence`): mention weight (log-scaled), source diversity, verified source proportion. Applied to `b2b_displacement_edges` (Sprint 1), `b2b_vendor_pain_points` (Sprint 2), `b2b_vendor_use_cases`, `b2b_vendor_integrations`, `b2b_vendor_buyer_profiles` (Sprint 4, migration 102). All tables have `min_confidence` filter on MCP tools and dashboard endpoints. | Not yet on `b2b_company_signals`, `b2b_churn_signals`, or `b2b_product_profiles`. Not propagated through full aggregation chain. |

**Phase 2 Sprint 1** (displacement edges + company signals): Migration 099 applied, live-validated.

**Phase 2 Sprint 2** (pain points + use cases + integrations): Migration 100 applied. Three provenance fetchers wired into `asyncio.gather` (20->23 concurrent). Three UPSERT persistence blocks with confidence scoring on pain points. Three new MCP tools (`list_vendor_pain_points`, `list_vendor_use_cases`, `list_vendor_integrations`). Three new dashboard endpoints (`/vendor-pain-points`, `/vendor-use-cases`, `/vendor-integrations`). CHECK constraint on pain_category matches actual enrichment values.

**Phase 2 Sprint 3** (buyer profiles): Migration 101 applied. `b2b_vendor_buyer_profiles` table with UPSERT per (vendor, role_type, buying_stage). Provenance fetcher, persistence block, MCP tool (`list_vendor_buyer_profiles`), dashboard endpoint (`/vendor-buyer-profiles`).

**Phase 2 Sprint 4** (confidence scoring close-out): Migration 102 adds `confidence_score NUMERIC(3,2)` to `b2b_vendor_use_cases`, `b2b_vendor_integrations`, `b2b_vendor_buyer_profiles`. `_compute_evidence_confidence()` wired into all three persistence blocks. `min_confidence` filter added to all three MCP tools and dashboard endpoints. Entity graph edges (`vendor -> strong_in_use_case`, `vendor -> weak_on_pain_category`) are satisfied by existing tables -- the rows ARE the edges.

Phase 2 deferred scope:
- Reviewer persona aggregations (low ROI vs. buyer profiles already captured)
- Company-level identity resolution (fuzzy matching, aliases)

### Phase 3: Historical Memory

| Item | Status | What's There | Key Gaps |
|------|--------|-------------|----------|
| Snapshots | **DONE** | `b2b_vendor_snapshots` table: daily append-only vendor health snapshots (churn density, urgency, recommend ratio, pain/competitor counts, displacement edges, high-intent companies). UPSERT per vendor per day. Configurable retention (default 365 days). Keyword signals and displacement edges also time-series. | `b2b_product_profiles` still upsert-overwritten (low ROI for snapshots). |
| Change events | **DONE** | `b2b_change_events` table: structural change event log (urgency spikes/drops, churn density spikes, NPS shifts, new pain categories, new competitors, review volume spikes). Detected by comparing today's data against prior snapshot. Append-only, configurable retention. | No pressure_score_spike or dm_churn_spike detection yet (deferred). |
| Historical queries | **DONE** | `get_vendor_history` MCP tool + dashboard endpoint for time-series queries. `list_change_events` MCP tool + dashboard endpoint with type/vendor filters. `compare_vendor_periods` MCP tool for two-date comparison. `change-events/summary` dashboard endpoint for aggregated view. Prior tools (displacement history, keyword rolling averages) still available. | No cross-vendor trend correlation queries. |

### Phase 4: Action Feedback Loop

| Item | Status | What's There | Key Gaps |
|------|--------|-------------|----------|
| Campaign outcome tracking | **DONE** (Sprint 1) | Full ESP webhook ingestion (opened, clicked, bounced, complained). Reply detection with intent classification. `campaign_funnel_stats` materialized view. Campaign audit log. **Sprint 1:** Outcome columns on `campaign_sequences` (migration 104). REST + MCP outcome recording. Signal effectiveness analysis (group by buying_stage, role_type, target_mode, score bucket, pain category). `outcome_history` JSONB tracks stage progression. Audit log `outcome_*` events. | Score calibration from outcomes (Sprint 2). CRM event ingestion (Sprint 3). |
| CRM event ingestion | **MISSING** | -- | No pipeline reads CRM events back into intelligence. No external CRM integration (HubSpot, Salesforce, Pipedrive). |
| Score calibration | **DONE** (Sprint 2) | `score_calibration_weights` table (migration 106). Weekly `b2b_score_calibration` autonomous task computes per-dimension conversion rates from outcomes, derives lift and weight adjustments. `_compute_score()` in campaign generation blends static defaults with calibrated weights (module-level cache, 1h TTL). MCP tools: `get_calibration_weights`, `trigger_score_calibration`. Dashboard: `GET /calibration-weights`. Capped at +/- 50% of static default to prevent wild swings. Requires 20+ sequences with outcomes before producing weights. | A/B testing infrastructure (deferred). |

### Phase 5: Thin Delivery Surfaces

| Item | Status | What's There | Key Gaps |
|------|--------|-------------|----------|
| API-first intelligence | **EXISTS** | Full REST API (dashboard, tenant, campaigns). B2B Churn MCP: 29 tools (was 25, +3 vendor entity tools in Sprint 2, +1 buyer profiles in Sprint 3). Intelligence MCP: 17 tools. All vendor entity tools support `min_confidence` filter (Sprint 4). | -- |
| Alternate delivery | **PARTIAL** | Email digest (weekly tenant reports via Resend). CSV export (signals, reviews, high-intent). ntfy push notifications. **Sprint 1:** Webhook outbound delivery (migration 107). `b2b_webhook_subscriptions` + `b2b_webhook_delivery_log` tables. Per-tenant webhook CRUD (6 REST endpoints). HMAC-SHA256 payload signing. Retry with backoff. Delivery log with success/failure tracking. Wired into change event detection and churn alert pipeline. MCP tools: `list_webhook_subscriptions`, `send_test_webhook_tool`. Config: `B2BWebhookConfig` (enable, timeout, retries). | No PDF export. No CRM sync. No Slack/Teams integration. |
| UI logic leakage | **CLEAN** | Frontend is display-only. All scoring, aggregation, analysis happen server-side. | -- |

### Phase 6: Analyst Controls

| Item | Status | What's There | Key Gaps |
|------|--------|-------------|----------|
| Correction tools | **PARTIAL** | Campaign suppression management (CRUD). Campaign review queue (bulk approve/reject). Scrape target management (CRUD + MCP). Blog post admin (draft/edit/publish). Vendor alias management (MCP: `add_vendor_alias`, `add_vendor_to_registry`). Vendor merge execution (Sprint 3): `merge_vendor` correction type renames vendor across 17 tables in a transaction, adds old name as alias, invalidates cache. | No source suppression/quality control. No review-level correction/flagging beyond suppress. |
| Correction persistence | **DONE** (Sprints 1+2) | `campaign_audit_log` (immutable campaign events). `campaign_suppressions` (manual overrides). `data_corrections` table (general-purpose, 8 entity types, 5 correction types, full audit trail). REST CRUD endpoints (4). MCP tools (3). Sprint 2: all downstream SELECT queries filter by active suppress corrections via NOT EXISTS subqueries (16 review fetchers, 15 dashboard queries, 12 MCP tool queries). | Vendor merge application, field-level override reads. |

### Overall Score

| Phase | Score | Summary |
|-------|-------|---------|
| Phase 0 | **90%** | Canonical source enum, vendor registry, provenance fields, source health metrics all delivered. Ingest-time normalization is the remaining gap. |
| Phase 1 | **90%** | Orchestration, telemetry, capability profiles, parser versioning all exist. Minor telemetry gaps (proxy IP, CAPTCHA solve time). |
| Phase 2 | **95%** | Sprints 1-4 complete: displacement edges, company signals, pain points, use cases, integrations, buyer profiles are first-class tables. Confidence scoring on all derived tables (edges, pain points, use cases, integrations, buyer profiles). Identity resolution still partial. Reviewer personas deferred (low ROI). |
| Phase 3 | **85%** | Daily vendor health snapshots, structural change event detection, historical query tools (MCP + dashboard). Displacement edges and keyword signals also time-series. Missing: product profile snapshots, cross-vendor correlation. |
| Phase 4 | **70%** | Sprints 1-2 complete: outcome tracking, signal effectiveness, score calibration from outcomes. CRM event ingestion remains. |
| Phase 5 | **80%** | API-first is solid (28 + 17 = 45 MCP tools). Sprint 1: webhook outbound delivery with per-tenant subscriptions, HMAC signing, retry, delivery log. Missing PDF export and CRM sync. UI is already clean. |
| Phase 6 | **75%** | Campaign-side + vendor alias controls exist. Sprint 1: `data_corrections` table, REST CRUD, MCP tools, audit trail. Sprint 2: correction-aware queries on all entity tables. Sprint 3: vendor merge execution across 17 tables with alias registration. |

### What's Strong Today

1. **Source orchestration** -- priority scheduling, retry, cooldown, concurrency, proxy fallback
2. **Scrape telemetry** -- every run logged with status, duration, proxy type, yield, parser version
3. **API-first delivery** -- full REST + 45 MCP tools, no intelligence logic in frontend
4. **Campaign lifecycle** -- ESP webhooks, reply detection, audit log, funnel analytics
5. **Consumer brand normalization** -- `comparisons.py` pipeline is production-grade
6. **Canonical source identity** -- `ReviewSource` enum, display names, classification sets used everywhere
7. **Vendor identity** -- DB-backed registry with aliases, MCP management tools

### Biggest Structural Gaps

1. **No feedback loop** -- campaign outcomes never improve signal scoring
2. **No company identity resolution** -- company names are free-text, no alias/merge system

### Testing checkpoint

**Sprint 1** (displacement edges + company signals): Validated -- migration 099 applied, tables populated.

**Sprints 2-4** (pain points, use cases, integrations, buyer profiles, confidence scoring): Migrations 100-102 applied.

1. Trigger a manual intelligence run and verify:
   - `pain_points_persisted > 0`, `use_cases_persisted > 0`, `integrations_persisted > 0`, `buyer_profiles_persisted > 0` in return dict
   - Confidence scores in 0.0-1.0 range on all tables
   - `source_distribution` JSONB populated on all tables
   - `sample_review_ids` reference real `b2b_reviews` rows
2. Test MCP tools: `list_vendor_pain_points`, `list_vendor_use_cases`, `list_vendor_integrations`, `list_vendor_buyer_profiles` (all support `min_confidence` filter)
3. Test dashboard endpoints: `GET /b2b/dashboard/vendor-pain-points`, `GET /b2b/dashboard/vendor-use-cases`, `GET /b2b/dashboard/vendor-integrations`, `GET /b2b/dashboard/vendor-buyer-profiles` (all support `min_confidence` filter)
4. Verify existing JSONB columns (`top_pain_categories`, `top_use_cases`, `top_integration_stacks`) still written (backwards compatible)
5. Verify all tests pass: `python -m pytest tests/test_b2b_intelligence_validation.py -x`

**Phase 4 Sprint 1** (campaign outcome tracking + signal effectiveness): Migration 104 applied.

1. Verify columns: `\d campaign_sequences` shows outcome, outcome_recorded_at, outcome_recorded_by, outcome_notes, outcome_revenue, outcome_history
2. Test REST: `POST /b2b/campaigns/sequences/{id}/outcome` with `{"outcome": "meeting_booked"}`, verify 200 + audit log entry
3. Test REST: `GET /b2b/campaigns/sequences/{id}/outcome` returns outcome + history
4. Test MCP: `record_campaign_outcome` tool records outcome and returns success
5. Test MCP: `get_signal_effectiveness` tool with group_by options (buying_stage, role_type, etc.)
6. Test dashboard: `GET /b2b/dashboard/signal-effectiveness` with tenant scoping
7. Verify outcome_history JSONB accumulates entries on repeated updates
8. Verify `campaign_audit_log` has `outcome_*` event types

**Phase 6 Sprint 1** (data corrections infrastructure): Migration 105 applied.

1. Verify table: `\d data_corrections` shows all columns, constraints (chk_correction_type, chk_correction_status, chk_entity_type), indexes
2. Test REST: `POST /b2b/dashboard/corrections` with `{"entity_type": "review", "entity_id": "<uuid>", "correction_type": "suppress", "reason": "Bad data"}`, verify 200
3. Test REST: `GET /b2b/dashboard/corrections` with filters (entity_type, correction_type, status), verify filtered results
4. Test REST: `GET /b2b/dashboard/corrections/{id}` returns full correction with metadata
5. Test REST: `POST /b2b/dashboard/corrections/{id}/revert` sets status=reverted, populates reverted_at/reverted_by
6. Test MCP: `create_data_correction` tool records correction and returns success
7. Test MCP: `list_data_corrections` tool with filters matches REST output
8. Test MCP: `revert_data_correction` tool reverts applied correction, rejects non-applied
9. Verify validation: invalid entity_type, correction_type, missing required fields all return errors
10. Verify corrected_by: REST sets `api:{user_id}` when authed, `analyst` when not; MCP defaults to `mcp`

**Phase 6 Sprint 2** (correction application logic): NOT EXISTS subqueries added to all read paths.

1. Create a suppress correction: `POST /b2b/dashboard/corrections` with entity_type=review, entity_id=<uuid>, correction_type=suppress
2. Verify `GET /b2b/dashboard/reviews` no longer returns the suppressed review
3. Verify `GET /b2b/dashboard/reviews/{id}` for the suppressed review returns 404
4. Verify `GET /b2b/dashboard/high-intent` excludes the suppressed review
5. Verify `GET /b2b/dashboard/signals` excludes suppressed churn signals
6. Verify MCP `search_reviews` excludes suppressed reviews
7. Verify MCP `get_review` for suppressed review returns "Review not found"
8. Verify MCP `get_pipeline_status` counts exclude suppressed reviews
9. Revert the correction: `POST /b2b/dashboard/corrections/{id}/revert`
10. Verify the review reappears in all queries after revert

**Phase 6 Sprint 3** (vendor merge execution): merge_vendor correction type now executes real merges.

1. Create merge_vendor correction: `POST /b2b/dashboard/corrections` with entity_type=vendor, entity_id=<any uuid>, correction_type=merge_vendor, old_value="Source Vendor", new_value="Target Vendor", reason="Duplicate"
2. Verify affected_count is populated on the correction record
3. Verify metadata contains per-table counts (table_counts)
4. Verify `b2b_reviews` rows with old vendor_name now show target vendor_name
5. Verify `b2b_vendors` target entry has old name as alias
6. Verify vendor cache invalidated (resolve_vendor_name returns canonical for old name)
7. Test MCP: `create_data_correction` with correction_type=merge_vendor returns merge info
8. Test validation: merge_vendor without old_value/new_value returns error

**Phase 4 Sprint 2** (score calibration from outcomes): Migration 106 applied.

1. Verify table: `\d score_calibration_weights` shows all columns, UNIQUE constraint, CHECK on dimension
2. Verify autonomous task registered: `b2b_score_calibration` in scheduler (cron: `0 4 * * 0` = Sunday 4AM)
3. Test MCP: `trigger_score_calibration` with insufficient data returns `calibrated: false` with reason
4. Record 20+ outcomes on sequences, then trigger calibration -- verify weights populated
5. Test MCP: `get_calibration_weights` returns weights grouped by dimension with lift values
6. Test dashboard: `GET /b2b/dashboard/calibration-weights` returns same data with optional dimension filter
7. Verify `_compute_score()` loads weights from cache and adjusts scoring (log before/after for a known row)
8. Verify weight_adjustment capped at +/- 50% of static_default (no wild swings)
9. Verify model_version increments on each calibration run

**Phase 5 Sprint 1** (webhook outbound delivery): Migration 107 applied.

1. Verify tables: `\d b2b_webhook_subscriptions` shows all columns, UNIQUE constraint on (account_id, url)
2. Verify tables: `\d b2b_webhook_delivery_log` shows all columns, FK to subscriptions
3. Test REST: `POST /b2b/dashboard/webhooks` creates subscription (requires auth), verify 200
4. Test REST: `GET /b2b/dashboard/webhooks` lists account's subscriptions
5. Test REST: `GET /b2b/dashboard/webhooks/{id}` returns subscription details
6. Test REST: `DELETE /b2b/dashboard/webhooks/{id}` deletes subscription + cascade delivery logs
7. Test REST: `GET /b2b/dashboard/webhooks/{id}/deliveries` returns delivery log
8. Test REST: `POST /b2b/dashboard/webhooks/{id}/test` sends test payload and returns success/failure
9. Test MCP: `list_webhook_subscriptions` returns subscriptions with 7-day delivery stats
10. Test MCP: `send_test_webhook_tool` delivers test payload to subscription URL
11. Verify HMAC signing: `X-Atlas-Signature` header matches `sha256=HMAC(secret, payload)`
12. Verify webhook dispatch wired into `_detect_change_events()` in b2b_churn_intelligence.py
13. Verify webhook dispatch wired into churn alert pipeline in b2b_churn_alert.py
14. Verify config: `ATLAS_B2B_WEBHOOK_ENABLED=true` enables delivery, `false` skips
15. Verify validation: invalid event_types rejected, empty event_types rejected, duplicate URL rejected (409)

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

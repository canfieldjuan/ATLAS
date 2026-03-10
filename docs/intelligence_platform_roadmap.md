# Intelligence Platform Roadmap

## Current-State Assessment (March 2026)

Audit performed against the live codebase. Each phase item rated as **EXISTS**, **PARTIAL**, or **MISSING**.

### Phase 0: Current-State Hardening -- COMPLETE

| Item | Status | What's There | Remaining Gaps |
|------|--------|-------------|----------|
| Source naming | **EXISTS** | `ReviewSource` enum (17 members) in `sources.py`. Classification sets (`VERIFIED_SOURCES`, `SLUG_SOURCES`, `SEARCH_SOURCES`, etc.), `display_name()` helper, `parse_source_allowlist()`. All consumers migrated. | -- |
| Vendor normalization | **EXISTS** | `b2b_vendor_registry` table (migration 095) with canonical_name + aliases array. `resolve_vendor_name()` at all 3 ingest points (scrape intake, REST, batch import). `resolve_vendor_name_cached()` at synthesis. MCP tools: `list_vendors_registry`, `add_vendor_to_registry`, `add_vendor_alias`. **Sprint 1:** Dedup key uses canonical vendor_name. **Sprint 2:** Fuzzy matching via pg_trgm (migration 114) + Python difflib fallback (0.85 threshold). Fuzzy vendor + company search endpoints and MCP tools. | -- |
| Provenance fields | **EXISTS** | Migration 096 adds `source_review_count`, `source_distribution` to `b2b_intelligence` and `b2b_churn_signals`. Intelligence reports and churn signals now back-reference source data. | No enrichment model/version tracking on individual reviews (see Phase 1 parser versioning). |
| Source reliability | **EXISTS** | Migration 097 adds `idx_scrape_log_source_health` composite index. `get_source_health` MCP tool + `/source-health` dashboard endpoint with per-source success rate, block rate, yield metrics, trend comparison. **Surface sprint:** `GET /source-health/telemetry` (CAPTCHA attempts, solve times, block type distribution, proxy usage per source). `GET /source-capabilities` REST wrapper. `GET /source-health/telemetry-timeline` (daily time-series). MCP: `get_source_telemetry`. | No automated alerting on degradation. |

### Phase 1: Managed Intelligence Substrate -- COMPLETE (100%)

| Item | Status | What's There | Remaining Gaps |
|------|--------|-------------|----------|
| Source orchestration | **EXISTS** | Priority-ordered scheduling, per-source semaphores (4 web / 10 API), retry with exponential backoff, cooldown after blocking, fallback proxy rotation, G2 3-tier fallback (Web Unlocker -> Playwright -> residential). | -- |
| Source capability profiles | **EXISTS** | `capabilities.py` module with `SourceCapabilityProfile` model. `get_source_capabilities` MCP tool + dashboard endpoint. Per-source: access pattern, anti-bot classification, proxy requirements, data quality tier. | -- |
| Scrape telemetry | **EXISTS** | `b2b_scrape_log` table: target_id, source, status, reviews_found/inserted, pages_scraped, errors (JSONB), duration_ms, proxy_type, started_at. **Sprint 1:** `captcha_attempts`, `captcha_types` (TEXT[]), `captcha_solve_ms`, `block_type` columns (migration 111). Block type classification (captcha/ip_ban/rate_limit/waf/unknown). Client-side CAPTCHA telemetry accumulation threaded through ScrapeResult to log. | -- |
| Parser versioning | **EXISTS** | Migration 098 adds `parser_version` to `b2b_scrape_log`. Each parser reports its version string. Enables selective re-extraction when parsers improve. **Sprint 2:** `_queue_version_upgrades()` in b2b_enrichment.py compares each review's `parser_version` against current parser version, resets outdated reviews to `enrichment_status='pending'` for automatic re-enrichment. Runs every enrichment cycle. REST: `GET /parser-version-status` shows per-source version status. MCP: `get_parser_version_status` tool. | -- |

### Phase 2: Canonical Intelligence Model -- COMPLETE

| Item | Status | What's There | Remaining Gaps |
|------|--------|-------------|----------|
| Canonical entities | **EXISTS** | `b2b_churn_signals` (per-vendor), `b2b_product_profiles` (vendor knowledge cards), `b2b_keyword_signals` (search volume), `b2b_alert_baselines`. **Sprint 1:** `b2b_displacement_edges` (append-only time-series, migration 099), `b2b_company_signals` (UPSERT per company-vendor, migration 099). **Sprint 2:** `b2b_vendor_pain_points` (UPSERT per vendor-category with confidence scoring, migration 100), `b2b_vendor_use_cases` (UPSERT per vendor-module, migration 100), `b2b_vendor_integrations` (UPSERT per vendor-tool, migration 100). **Sprint 3:** `b2b_vendor_buyer_profiles` (UPSERT per vendor-role-stage, migration 101). | Reviewer personas still in JSONB (deferred -- low ROI vs. buyer profiles). |
| Identity resolution | **EXISTS** | DB-backed `b2b_vendor_registry` with aliases (Phase 0). `_canonicalize_vendor()` / `_canonicalize_competitor()` applied at synthesis. Review-level dedup via SHA-256 dedup_key. **Phase 6 Sprint 3:** Vendor merge execution across 17 tables. **Phase 0/2 Sprint:** Fuzzy matching via pg_trgm trigram similarity (migration 114) + Python difflib fallback. Vendor + company fuzzy search (REST + MCP). | -- |
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
| Snapshots | **DONE** | `b2b_vendor_snapshots` table: daily append-only vendor health snapshots (churn density, urgency, recommend ratio, pain/competitor counts, displacement edges, high-intent companies). UPSERT per vendor per day. Configurable retention (default 365 days). Keyword signals and displacement edges also time-series. **Sprint 5:** `b2b_product_profile_snapshots` table (migration 115): daily product profile snapshots (review counts, ratings, strength/weakness counts, top items, competitive positioning counts). Captured automatically after profile generation. REST: `GET /product-profile-history`. MCP: `get_product_profile_history`. | -- |
| Change events | **DONE** | `b2b_change_events` table: structural change event log (urgency spikes/drops, churn density spikes, NPS shifts, new pain categories, new competitors, review volume spikes, pressure_score_spike, dm_churn_spike). Detected by comparing today's data against prior snapshot. Append-only, configurable retention. Migration 117 adds `pressure_score`, `dm_churn_rate`, `price_complaint_rate` to `b2b_vendor_snapshots`. Pressure spike threshold: 10 pts (0-100). DM churn spike threshold: 15 pp. | -- |
| Historical queries | **DONE** | `get_vendor_history` MCP tool + dashboard endpoint for time-series queries. `list_change_events` MCP tool + dashboard endpoint with type/vendor filters. `compare_vendor_periods` MCP tool for two-date comparison. `change-events/summary` dashboard endpoint for aggregated view. Prior tools (displacement history, keyword rolling averages) still available. **Sprint 4:** Cross-vendor trend correlation. `GET /concurrent-events` finds dates where 3+ vendors had the same event type (market-level signals). `GET /vendor-correlation` computes pairwise Pearson r over aligned snapshot time-series with displacement edge context. MCP tools: `list_concurrent_events`, `get_vendor_correlation`. Automatic `concurrent_shift` event detection in nightly intelligence run. **Sprint 5:** Product profile history queries via `get_product_profile_history` MCP tool + `GET /product-profile-history` dashboard endpoint. | -- |

### Phase 4: Action Feedback Loop

| Item | Status | What's There | Key Gaps |
|------|--------|-------------|----------|
| Campaign outcome tracking | **DONE** (Sprint 1) | Full ESP webhook ingestion (opened, clicked, bounced, complained). Reply detection with intent classification. `campaign_funnel_stats` materialized view. Campaign audit log. **Sprint 1:** Outcome columns on `campaign_sequences` (migration 104). REST + MCP outcome recording. Signal effectiveness analysis (group by buying_stage, role_type, target_mode, score bucket, pain category). `outcome_history` JSONB tracks stage progression. Audit log `outcome_*` events. | Score calibration from outcomes (Sprint 2). CRM event ingestion (Sprint 3). |
| CRM event ingestion | **DONE** (Sprints 3-5) | `b2b_crm_events` table (migration 108) with status tracking, dedup, partial unique index. Inbound webhooks: generic (`POST /b2b/crm/events`), batch (`POST /b2b/crm/events/batch`), HubSpot-native (`POST /b2b/crm/events/hubspot`), **Salesforce-native** (`POST /b2b/crm/events/salesforce`), **Pipedrive-native** (`POST /b2b/crm/events/pipedrive`). Autonomous `crm_event_processing` task (5-min interval) matches events to campaign sequences by email/company, auto-records outcomes with rank-based progression (prevents downgrades). All three CRM webhook formats normalize to internal event schema. Config: `CRMEventConfig` (enabled, batch_size, stage_outcome_map). MCP tools: `list_crm_events`, `ingest_crm_event`. **Sprint 5:** Event enrichment -- cross-event field resolution (deal_id siblings), vendor name normalization via `resolve_vendor_name()`, fuzzy company matching (pg_trgm similarity >= 0.6) as fallback. Enriched fields persisted back to event row. REST: `GET /events/enrichment-stats`. MCP: `get_crm_enrichment_stats`. | -- |
| Score calibration | **DONE** (Sprint 2) | `score_calibration_weights` table (migration 106). Weekly `b2b_score_calibration` autonomous task computes per-dimension conversion rates from outcomes, derives lift and weight adjustments. `_compute_score()` in campaign generation blends static defaults with calibrated weights (module-level cache, 1h TTL). MCP tools: `get_calibration_weights`, `trigger_score_calibration`. Dashboard: `GET /calibration-weights`. Capped at +/- 50% of static default to prevent wild swings. Requires 20+ sequences with outcomes before producing weights. | A/B testing infrastructure (deferred -- low ROI until volume justifies). |

### Phase 5: Thin Delivery Surfaces

| Item | Status | What's There | Key Gaps |
|------|--------|-------------|----------|
| API-first intelligence | **EXISTS** | Full REST API (dashboard, tenant, campaigns). B2B Churn MCP: 35+ tools. Intelligence MCP: 17 tools. All vendor entity tools support `min_confidence` filter. **Surface sprint:** REST gaps closed -- `GET /product-profile`, `GET /displacement-history`, `GET /source-health/telemetry`, `GET /source-health/telemetry-timeline`, `GET /source-capabilities`, `GET /operational-overview`. MCP: `get_source_telemetry`, `get_operational_overview`. | -- |
| Alternate delivery | **EXISTS** | Email digest (weekly tenant reports via Resend). CSV export (signals, reviews, high-intent). ntfy push notifications. **Sprint 1:** Webhook outbound delivery (migration 107). Per-tenant webhook CRUD, HMAC-SHA256 signing, retry with backoff, delivery logging. MCP tools: `list_webhook_subscriptions`, `send_test_webhook_tool`. **Sprint 2:** PDF intelligence report export (`fpdf2`). MCP tool: `export_report_pdf`. **Sprint 3:** Slack + Teams notification channels (migration 112). Slack Block Kit + Teams Adaptive Card formatters. **Sprint 4:** CRM outbound push (migration 113). `crm_hubspot`/`crm_salesforce`/`crm_pipedrive` webhook channels. CRM-specific payload formatting (HubSpot properties, Salesforce custom fields, Pipedrive deals). `auth_header` column for CRM API auth. `b2b_crm_push_log` table tracks push history. REST endpoint: `GET /webhooks/{id}/crm-push-log`. MCP tool: `list_crm_pushes`. | -- |
| UI logic leakage | **CLEAN** | Frontend is display-only. All scoring, aggregation, analysis happen server-side. | -- |

### Phase 6: Analyst Controls

| Item | Status | What's There | Key Gaps |
|------|--------|-------------|----------|
| Correction tools | **DONE** | Campaign suppression management (CRUD). Campaign review queue (bulk approve/reject). Scrape target management (CRUD + MCP). Blog post admin (draft/edit/publish). Vendor alias management (MCP: `add_vendor_alias`, `add_vendor_to_registry`). Vendor merge execution (Sprint 3): `merge_vendor` correction type renames vendor across 17 tables in a transaction, adds old name as alias, invalidates cache. Sprint 5: source quality controls -- `suppress_source` correction type suppresses all reviews from a source (globally or vendor-scoped), impact reporting endpoint. | -- |
| Correction persistence | **DONE** (Sprints 1-5) | `campaign_audit_log` (immutable campaign events). `campaign_suppressions` (manual overrides). `data_corrections` table (general-purpose, 9 entity types, 6 correction types, full audit trail). REST CRUD endpoints (5). MCP tools (4). Sprint 2: all downstream SELECT queries filter by active suppress corrections via NOT EXISTS subqueries (16 review fetchers, 15 dashboard queries, 12 MCP tool queries). Sprint 4: field override reads on single-entity endpoints. Sprint 5: source suppression wired into `_eligible_review_filters()` for intelligence pipeline. | -- |

### Overall Score

| Phase | Score | Summary |
|-------|-------|---------|
| Phase 0 | **100%** | Canonical source enum, vendor registry, provenance fields, source health metrics all delivered. Ingest-time vendor normalization live. Sprint 1: dedup key uses canonical vendor_name. Sprint 2: fuzzy vendor name matching -- pg_trgm extension (migration 114), trigram similarity search on `b2b_vendors` + `b2b_company_signals`, Python-level fuzzy fallback in `_resolve_from_cache()` (difflib SequenceMatcher, 0.85 threshold). REST endpoints: `GET /fuzzy-vendor-search`, `GET /fuzzy-company-search`. MCP tools: `fuzzy_vendor_search`, `fuzzy_company_search`. |
| Phase 1 | **100%** | Orchestration, telemetry, capability profiles, parser versioning all exist. Sprint 1: CAPTCHA solve time, block type classification, captcha_types array persisted (migration 111). Sprint 2: automatic re-processing on parser version change (`_queue_version_upgrades()` resets outdated reviews to pending). REST: `GET /parser-version-status`. MCP: `get_parser_version_status`. |
| Phase 2 | **100%** | Sprints 1-4 complete: displacement edges, company signals, pain points, use cases, integrations, buyer profiles. Confidence scoring on all derived tables. Sprint 5: fuzzy identity resolution -- vendor + company fuzzy matching via pg_trgm (migration 114), Python-level fuzzy fallback in vendor cache. Reviewer personas deferred (low ROI). |
| Phase 3 | **100%** | Sprints 1-5 complete: daily snapshots, change event detection, historical queries, cross-vendor trend correlation (concurrent events, pairwise Pearson r, automatic concurrent_shift detection). Sprint 5: product profile snapshots (migration 115, daily captures after profile generation). REST: `GET /product-profile-history`. MCP: `get_product_profile_history`. |
| Phase 4 | **100%** | Sprints 1-5 complete: outcome tracking + signal effectiveness (Sprint 1), score calibration from outcomes (Sprint 2), CRM event ingestion pipeline (Sprint 3), Salesforce + Pipedrive native webhooks (Sprint 4), CRM event enrichment (Sprint 5, cross-event resolution, vendor normalization, fuzzy matching). A/B testing deferred (low ROI until volume justifies). |
| Phase 5 | **100%** | API-first solid (45+ MCP tools). Sprint 1: webhook outbound delivery. Sprint 2: PDF report export. Sprint 3: Slack + Teams (Block Kit, Adaptive Cards). Sprint 4: CRM outbound push (HubSpot, Salesforce, Pipedrive). All delivery surfaces covered: API, webhooks, email, PDF, Slack, Teams, CRM push, ntfy, CSV. |
| Phase 6 | **100%** | Sprints 1-5 complete. Sprint 1: data_corrections table + CRUD + MCP + audit. Sprint 2: correction-aware queries on all entity tables. Sprint 3: vendor merge across 20 tables (was 17; added b2b_scrape_targets, b2b_product_profile_snapshots, campaign_funnel_stats refresh). Sprint 4: field override reads. Sprint 5: source quality controls (suppress_source, impact). Correction logic (suppress_predicate, apply_field_overrides) extracted to shared `services/b2b/corrections.py`. |

### What's Strong Today

1. **Source orchestration** -- priority scheduling, retry, cooldown, concurrency, proxy fallback
2. **Scrape telemetry** -- every run logged with status, duration, proxy type, yield, parser version
3. **API-first delivery** -- full REST + 45 MCP tools, no intelligence logic in frontend
4. **Campaign lifecycle** -- ESP webhooks, reply detection, audit log, funnel analytics
5. **Consumer brand normalization** -- `comparisons.py` pipeline is production-grade
6. **Canonical source identity** -- `ReviewSource` enum, display names, classification sets used everywhere
7. **Vendor identity** -- DB-backed registry with aliases, MCP management tools

### Consumer Intelligence Infrastructure Port

B2B intelligence infrastructure ported to consumer product review pipeline (migrations 121-122):

| Capability | Migration/File | Status |
|---|---|---|
| Brand snapshots | `121_consumer_snapshots.sql` -- `brand_intelligence_snapshots` table | **DONE** |
| Change event detection | `121_consumer_snapshots.sql` -- `product_change_events` table | **DONE** |
| Consumer corrections | `122_consumer_corrections.sql` -- extended `data_corrections` CHECK | **DONE** |
| Snapshot persistence | `competitive_intelligence.py` -- `_persist_brand_snapshots()` | **DONE** |
| Change event detection | `competitive_intelligence.py` -- `_detect_change_events()` | **DONE** |
| Correction-aware queries | `consumer_dashboard.py` -- `suppress_predicate` on reviews | **DONE** |
| Field override reads | `consumer_dashboard.py` -- `apply_field_overrides` on single review | **DONE** |
| Brand history endpoint | `GET /consumer/dashboard/brand-history` | **DONE** |
| Change events endpoint | `GET /consumer/dashboard/change-events` | **DONE** |
| Change events summary | `GET /consumer/dashboard/change-events/summary` | **DONE** |
| Corrections CRUD | `POST/GET /consumer/dashboard/corrections` + revert + stats | **DONE** |
| Export endpoints | `GET /consumer/dashboard/export/reviews,brands,pain-points` | **DONE** |
| MCP: brand history | `get_brand_history` tool | **DONE** |
| MCP: change events | `list_product_change_events` tool | **DONE** |
| MCP: corrections CRUD | `create/list/revert_consumer_correction` tools | **DONE** |

Consumer change event types: `pain_score_spike` (>= 1.5 pts), `vulnerability_spike` (>= 10 pts), `safety_flag_emergence` (0 to >0), `repurchase_decline` (>= 15 pp), `rating_drop` (>= 0.5 stars).

Intelligence MCP server expanded from 17 to 24 tools.
Consumer dashboard expanded from 14 to 24 endpoints.

### Biggest Structural Gaps

1. ~~**No feedback loop**~~ **CLOSED** -- Sprint 1 outcome tracking, Sprint 2 score calibration, Sprint 3 CRM event ingestion
2. ~~**No company identity resolution**~~ **CLOSED** -- vendor merge execution (Phase 6 Sprint 3) handles aliases + batch rename. Fuzzy matching via pg_trgm on vendors + companies (Phase 0/2 Sprint). Python-level fuzzy fallback in vendor cache.
3. ~~**No cross-vendor correlation**~~ **CLOSED** -- Sprint 4 adds concurrent event detection, pairwise Pearson correlation, and automatic `concurrent_shift` events
4. ~~**Consumer pipeline missing B2B infrastructure**~~ **CLOSED** -- Snapshots, change events, corrections, export, history endpoints all ported

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

**Phase 3 Sprint 4** (cross-vendor trend correlation): Migration 109 applied (3 correlation indexes).

1. Verify indexes: `\di idx_bce_date_type`, `\di idx_bvs_date_vendor`, `\di idx_displacement_edges_date_pair`
2. Test REST: `GET /b2b/dashboard/concurrent-events?days=30&min_vendors=2` returns dates with multi-vendor events
3. Test REST: `GET /b2b/dashboard/vendor-correlation?vendor_a=X&vendor_b=Y&metric=churn_density` returns time-series + Pearson r
4. Test REST: vendor-correlation with invalid metric returns 400
5. Test MCP: `list_concurrent_events` with event_type filter returns filtered concurrent shifts
6. Test MCP: `get_vendor_correlation` returns correlation, series, displacement edges
7. Verify concurrent_shift detection: insert 3+ change events for same event_type on same date, trigger intelligence run, verify `concurrent_shift` event logged for `__market__` vendor
8. Verify correlation direction: two vendors with inverse churn trends should show negative Pearson r

**Phase 3 Sprint 5** (product profile snapshots): Migration 115 applied.

1. Verify table: `\d b2b_product_profile_snapshots` shows vendor_name, snapshot_date, 14 metric columns, UNIQUE constraint
2. Verify indexes: `\di idx_bpps_vendor_date`, `\di idx_bpps_date`
3. Trigger product profile generation: run `b2b_product_profiles` task, verify `snapshots_persisted > 0` in return dict
4. Verify snapshots: `SELECT * FROM b2b_product_profile_snapshots LIMIT 5` shows daily snapshot rows
5. Test REST: `GET /b2b/dashboard/product-profile-history?vendor_name=Salesforce` returns snapshot time-series
6. Test MCP: `get_product_profile_history(vendor_name="Salesforce")` returns same data
7. Verify idempotent: re-run profile generation, verify ON CONFLICT updates existing snapshots (no duplicates)
8. AST parse: `python -c "import ast; ast.parse(open('atlas_brain/autonomous/tasks/b2b_product_profiles.py').read())"`

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

**Phase 6 Sprint 4** (field override reads): Single-entity endpoints return corrected values when override_field corrections exist.

1. Create override_field correction: `POST /b2b/dashboard/corrections` with entity_type=review, entity_id=<uuid>, correction_type=override_field, field_name=vendor_name, old_value="Original", new_value="Corrected", reason="typo"
2. Fetch the review: `GET /b2b/dashboard/reviews/<uuid>` -- verify vendor_name shows "Corrected" and `_overrides_applied` array is present
3. Fetch via MCP: `get_review` -- same verification
4. Create override on churn_signal: `GET /b2b/dashboard/signals/<vendor>` -- verify overridden field and `_overrides_applied`
5. Fetch vendor profile: `GET /b2b/dashboard/vendors/<vendor>` -- verify churn_signal sub-dict has overridden field
6. Revert the correction, fetch again -- verify original value restored and `_overrides_applied` absent
7. Verify list queries (GET /reviews, GET /signals) are NOT affected (overrides only on single-entity reads)

**Phase 6 Sprint 5** (source quality controls): Source-level suppression via data corrections.

1. Apply migration: `psql -p 5433 -f atlas_brain/storage/migrations/110_source_quality.sql`
2. Verify CHECK constraints updated: `\d data_corrections` shows entity_type includes 'source', correction_type includes 'suppress_source'
3. Create global source suppression: `POST /b2b/dashboard/corrections` with `{entity_type: "source", entity_id: "<uuid5>", correction_type: "suppress_source", metadata: {"source_name": "reddit"}, reason: "Low quality"}`
4. Create vendor-scoped suppression: same but add `field_name: "Acme Software"`
5. Verify intelligence queries exclude reviews from suppressed source (trigger intelligence run or inspect query)
6. Verify `GET /b2b/dashboard/source-corrections/impact` shows affected review counts
7. Verify MCP `get_source_correction_impact` returns same data
8. Verify MCP `create_data_correction` with suppress_source and metadata validates source_name against known sources
9. Revert the correction, verify reviews reappear in intelligence queries
10. Test validation: suppress_source without metadata.source_name returns error
11. Test validation: suppress_source with unknown source name returns error

**Phase 1 Sprint 1** (CAPTCHA telemetry + block type classification): Migration 111 applied.

1. Verify columns: `\d b2b_scrape_log` shows captcha_attempts (INT DEFAULT 0), captcha_types (TEXT[] DEFAULT '{}'), captcha_solve_ms (INT), block_type (TEXT)
2. Verify ScrapeResult dataclass: `python -c "from atlas_brain.services.scraping.parsers import ScrapeResult; r = ScrapeResult([], 0, [], captcha_attempts=1, captcha_types=['turnstile'], captcha_solve_ms=500); assert r.captcha_attempts == 1"`
3. Verify client tracking: `python -c "from atlas_brain.services.scraping.client import AntiDetectionClient; assert hasattr(AntiDetectionClient, 'reset_captcha_stats')"`
4. Verify block type classification: blocked+403 -> waf, 429+rate -> rate_limit, captcha in errors -> captcha
5. Verify autonomous scrape: trigger a scrape run, check `b2b_scrape_log` rows have captcha_attempts populated (0 if no CAPTCHA)
6. Verify API scrape: `POST /b2b/scrape/run-target` also writes captcha telemetry columns
7. AST parse all modified files: client.py, parsers/__init__.py, b2b_scrape_intake.py, b2b_scrape.py

**Phase 1 Sprint 2** (auto re-processing on parser version change): No migration required.

1. Verify `_queue_version_upgrades()` exists: `python -c "from atlas_brain.autonomous.tasks.b2b_enrichment import _queue_version_upgrades; print('OK')"`
2. Verify REST: `GET /b2b/dashboard/parser-version-status` returns per-source version status with review counts
3. Verify MCP: `get_parser_version_status` returns same data as REST endpoint
4. Verify re-queue logic: manually set a review's `parser_version` to an old value, trigger enrichment cycle, verify review reset to `enrichment_status='pending'`
5. Verify return dict: enrichment run return dict includes `version_upgrade_requeued` count when > 0
6. Verify fail-open: if `get_all_parsers()` raises, function returns 0 and enrichment proceeds normally
7. AST parse: `python -c "import ast; ast.parse(open('atlas_brain/autonomous/tasks/b2b_enrichment.py').read())"`

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

**Phase 4 Sprint 3** (CRM event ingestion pipeline): Migration 108 applied.

1. Verify table: `\d b2b_crm_events` shows all columns, status CHECK constraint, partial unique index on (crm_provider, crm_event_id)
2. Verify config: `CRMEventConfig` loads from env, `settings.crm_event.enabled` defaults to False
3. Test REST: `POST /b2b/crm/events` with valid payload returns event ID and status=pending
4. Test REST: `POST /b2b/crm/events` rejects invalid crm_provider or event_type (400)
5. Test REST: `POST /b2b/crm/events/batch` ingests multiple events, returns per-event errors
6. Test REST: `POST /b2b/crm/events/hubspot` normalizes HubSpot webhook format (deal.propertyChange -> deal_stage_change)
7. Test REST: `GET /b2b/crm/events` lists events with status/provider/company filters
8. Verify autonomous task: `crm_event_processing` registered in `_BUILTIN_TASKS` and `_DEFAULT_TASKS` (5-min interval)
9. Test processing: insert pending event with matching campaign sequence email, wait for task run, verify outcome recorded on sequence
10. Test rank guard: insert event that would downgrade outcome (e.g. meeting_booked after deal_won), verify event marked as skipped
11. Test MCP: `list_crm_events` returns filtered events, `ingest_crm_event` creates pending event
12. Verify dedup: POST same crm_event_id twice, verify ON CONFLICT updates rather than duplicates

**Phase 4 Sprint 4** (Salesforce + Pipedrive native webhooks): No new migration required.

1. Test Salesforce: `POST /b2b/crm/events/salesforce` with `{"sobject": "Opportunity", "record": {"Id": "006xxx", "StageName": "Closed Won", "Amount": 50000, "Account": {"Name": "Acme Corp"}, "Contact": {"Email": "j@acme.com"}, "Name": "Acme Deal"}}` -- verify event created with provider=salesforce, event_type=deal_won
2. Test Salesforce stage mapping: "Closed Lost" -> deal_lost, "Demo Scheduled" -> meeting_booked, "Qualification" -> deal_stage_change
3. Test Pipedrive: `POST /b2b/crm/events/pipedrive` with `{"event": "updated.deal", "current": {"id": 123, "title": "Big Deal", "status": "won", "org_name": "Acme", "value": 25000}}` -- verify event_type=deal_won
4. Test Pipedrive status mapping: "lost" -> deal_lost, activity with "meeting" -> meeting_booked
5. Test auth: both endpoints require authentication (401 without token)
6. Test dedup: POST same Salesforce/Pipedrive event twice, verify ON CONFLICT updates
7. Verify events appear in `GET /b2b/crm/events?crm_provider=salesforce` and `crm_provider=pipedrive`
8. Verify autonomous task processes Salesforce/Pipedrive events same as HubSpot (matches by email/company)

**Phase 4 Sprint 5** (CRM event enrichment): No migration required.

1. Verify enrichment function exists: `python -c "from atlas_brain.autonomous.tasks.crm_event_processing import _enrich_event_fields; print('OK')"`
2. Test cross-event enrichment: insert two events with same deal_id, one with company_name and one without. Run processing task, verify second event inherits company_name.
3. Test vendor normalization: insert event with company_name="salesforce", verify it's normalized to canonical "Salesforce" after processing
4. Test fuzzy matching: insert event with company_name="Salseforce" (typo), verify fuzzy match finds the correct campaign sequence (similarity >= 0.6)
5. Test REST: `GET /b2b/crm/events/enrichment-stats` returns field coverage + enrichment counts
6. Test MCP: `get_crm_enrichment_stats` returns same stats as REST endpoint
7. Verify enriched marker: processed events with enriched fields have `[enriched]` in processing_notes
8. AST parse: `python -c "import ast; ast.parse(open('atlas_brain/autonomous/tasks/crm_event_processing.py').read())"`

**Surface Sprint 4** (Phase 4 action feedback loop surfaces): No migration required.

1. Test REST: `GET /b2b/campaigns/sequences?outcome=deal_won` returns only sequences with outcome=deal_won
2. Test REST: `GET /b2b/campaigns/sequences?outcome=invalid` returns 400 with valid outcome list
3. Test REST: `GET /b2b/crm/events?status=matched&start_date=2026-01-01&end_date=2026-04-01` returns date-filtered events
4. Test REST: `GET /b2b/crm/events?status=invalid` returns 400 with valid status list
5. Test REST: `POST /b2b/crm/events/batch` returns `created_ids` array matching ingested count
6. Test REST: `GET /b2b/dashboard/outcome-distribution` returns funnel with counts, pct, revenue per outcome
7. Test REST: `GET /b2b/dashboard/signal-to-outcome?group_by=buying_stage` returns attribution groups
8. Test REST: `GET /b2b/dashboard/signal-to-outcome?group_by=invalid` returns 400
9. Test REST: `POST /b2b/dashboard/calibration/trigger` runs calibration and returns result
10. Test MCP: `list_crm_events(start_date="2026-01-01", end_date="2026-04-01")` returns date-filtered events
11. Test MCP: `list_crm_events(status="invalid")` returns error
12. Test MCP: `get_outcome_distribution` returns funnel view matching REST
13. Test MCP: `trigger_score_calibration(window_days=180)` runs calibration and returns weights
14. AST parse all modified files: b2b_campaigns.py, b2b_crm_events.py, b2b_dashboard.py, b2b_churn_server.py

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

**Phase 5 Sprint 2** (PDF intelligence report export): fpdf2 installed.

1. AST parse: `python -c "import ast; ast.parse(open('atlas_brain/services/b2b/pdf_renderer.py').read())"`
2. Import check: `python -c "from atlas_brain.services.b2b.pdf_renderer import render_report_pdf"`
3. Smoke test: generate PDFs for all 5 report families (churn feed, scorecard, comparison, deep dive, generic)
4. Verify PDF header: output starts with `%PDF`
5. Test REST: `GET /b2b/dashboard/reports/{id}/pdf` returns `application/pdf` with `Content-Disposition` header
6. Test access control: unauthenticated user can access, tracked-vendor check enforced
7. Test 404: nonexistent report_id returns 404
8. Test 400: invalid UUID returns 400
9. Test MCP: `export_report_pdf` returns JSON with `filename`, `size_bytes`, `content_base64`
10. Verify base64 decode: `base64.b64decode(content_base64)` produces valid PDF bytes
11. Non-ASCII check: `grep -rP '[\x80-\xFF]' atlas_brain/services/b2b/pdf_renderer.py` returns empty
12. Verify fpdf2 in requirements.txt

**Phase 5 Sprint 3** (Slack + Teams notification channels): Migration 112 applied.

1. Verify column: `\d b2b_webhook_subscriptions` shows `channel TEXT NOT NULL DEFAULT 'generic'` with CHECK constraint
2. Test REST: `POST /b2b/dashboard/webhooks` with `{url, secret, event_types, channel: "slack"}` creates Slack subscription
3. Test REST: `POST /b2b/dashboard/webhooks` with invalid channel (e.g. "discord") returns 400
4. Test REST: `GET /b2b/dashboard/webhooks` returns `channel` field on each subscription
5. Test REST: `GET /b2b/dashboard/webhooks/{id}` returns `channel` field
6. Test MCP: `list_webhook_subscriptions` returns `channel` on each subscription
7. Verify Slack formatting: trigger a change event for a tracked vendor with a Slack subscription, inspect delivery log payload for Block Kit structure
8. Verify Teams formatting: same test with a Teams subscription, inspect for Adaptive Card structure
9. Test test webhook: `POST /b2b/dashboard/webhooks/{id}/test` for Slack/Teams subscriptions sends correctly formatted payload
10. AST parse: `python -c "import ast; ast.parse(open('atlas_brain/services/b2b/webhook_dispatcher.py').read())"`

**Phase 5 Sprint 4** (CRM outbound push): Migration 113 applied.

1. Verify column: `\d b2b_webhook_subscriptions` shows `channel` CHECK includes crm_hubspot/crm_salesforce/crm_pipedrive, `auth_header TEXT` column
2. Verify table: `\d b2b_crm_push_log` shows signal_type, vendor_name, company_name, crm_record_id, crm_record_type, status
3. Test REST: `POST /b2b/dashboard/webhooks` with `{channel: "crm_hubspot", auth_header: "Bearer pat-xxx", ...}` creates CRM subscription
4. Test REST: `POST /b2b/dashboard/webhooks` with `{channel: "crm_hubspot"}` and no auth_header returns 400
5. Test REST: `GET /b2b/dashboard/webhooks/{id}/crm-push-log` returns push history
6. Test MCP: `list_crm_pushes` returns push log entries with channel and webhook_url
7. Verify HubSpot formatting: create crm_hubspot subscription, trigger churn_alert, inspect delivery log for HubSpot properties format
8. Verify Salesforce formatting: inspect for custom field format (`Atlas_Vendor__c`, etc.)
9. Verify Pipedrive formatting: inspect for deal title format
10. Verify push logging: after successful CRM delivery, `b2b_crm_push_log` has new row with signal_type and vendor_name
11. AST parse all modified files: webhook_dispatcher.py, b2b_dashboard.py, b2b_churn_server.py

**Phase 0/2 Sprint** (fuzzy vendor + company matching): Migration 114 applied (pg_trgm + 2 GiST indexes).

1. Verify extension: `SELECT * FROM pg_extension WHERE extname = 'pg_trgm'`
2. Verify indexes: `\di idx_b2b_vendors_canonical_trgm`, `\di idx_b2b_company_signals_company_trgm`
3. Test SQL similarity: `SELECT canonical_name, similarity(canonical_name, 'Salesfroce') FROM b2b_vendors WHERE similarity(canonical_name, 'Salesfroce') > 0.3 ORDER BY similarity DESC`
4. Test REST: `GET /b2b/dashboard/fuzzy-vendor-search?q=Salesfroce` returns "Salesforce" as top match
5. Test REST: `GET /b2b/dashboard/fuzzy-vendor-search?q=` returns 400
6. Test REST: `GET /b2b/dashboard/fuzzy-company-search?q=Acme&vendor_name=Salesforce` returns matching companies
7. Test MCP: `fuzzy_vendor_search(query="Hubspt")` returns "HubSpot" as top match
8. Test MCP: `fuzzy_company_search(query="Acm Corp")` returns similar company names
9. Test Python fuzzy fallback: `python -c "from atlas_brain.services.vendor_registry import _resolve_from_cache; ..."` with a near-miss name returns canonical match
10. Verify cache performance: fuzzy fallback only activates for inputs >= 4 chars (short strings bypass)
11. AST parse: `python -c "import ast; ast.parse(open('atlas_brain/services/vendor_registry.py').read())"`

**Surface Sprint 5** (Phase 5 thin delivery surfaces): No migration required.

1. Test REST: `PATCH /b2b/dashboard/webhooks/{id}` with `{"enabled": false}` disables webhook, returns updated record
2. Test REST: `PATCH /b2b/dashboard/webhooks/{id}` with `{"event_types": ["churn_alert"]}` updates event filter
3. Test REST: `PATCH /b2b/dashboard/webhooks/{id}` with invalid event_types returns 400
4. Test REST: `PATCH /b2b/dashboard/webhooks/{id}` with empty body returns 400 ("No fields to update")
5. Test REST: `GET /b2b/dashboard/webhooks` now includes `recent_deliveries_7d` and `recent_success_rate_7d` per webhook
6. Test REST: `GET /b2b/dashboard/webhooks/{id}/deliveries?success=false` returns only failed deliveries
7. Test REST: `GET /b2b/dashboard/webhooks/{id}/deliveries?start_date=2026-03-01&end_date=2026-03-10` returns date-filtered deliveries
8. Test REST: `GET /b2b/dashboard/webhooks/{id}/deliveries?event_type=churn_alert` returns event-type-filtered deliveries
9. Test REST: `GET /b2b/dashboard/webhooks/delivery-summary?days=7` returns aggregate delivery health (total, success rate, p95 latency)
10. Test MCP: `update_webhook(subscription_id="...", enabled=false)` toggles webhook off
11. Test MCP: `update_webhook(subscription_id="...", event_types="change_event,churn_alert")` updates event filter
12. Test MCP: `get_webhook_delivery_summary(days=7)` returns aggregate stats matching REST
13. AST parse all modified files: b2b_dashboard.py, b2b_churn_server.py

**Surface Sprint 6** (Phase 6 analyst control surfaces): No migration required.

1. Test REST: `GET /b2b/dashboard/corrections?corrected_by=api` returns only corrections by API users
2. Test REST: `GET /b2b/dashboard/corrections?start_date=2026-03-01&end_date=2026-03-10` returns date-filtered corrections
3. Test REST: `GET /b2b/dashboard/corrections?corrected_by=mcp&correction_type=suppress` combines filters correctly
4. Test REST: `GET /b2b/dashboard/corrections/stats?days=30` returns aggregate activity (by_status, by_type, by_entity, top_correctors)
5. Test REST: `GET /b2b/dashboard/corrections/stats?days=7` returns narrower window
6. Test MCP: `list_data_corrections(corrected_by="mcp")` returns filtered corrections
7. Test MCP: `list_data_corrections(start_date="2026-03-01", end_date="2026-03-10")` returns date-filtered
8. Test MCP: `get_data_correction(correction_id="...")` returns full correction details
9. Test MCP: `get_data_correction(correction_id="nonexistent-uuid")` returns "Correction not found"
10. Test MCP: `get_correction_stats(days=30)` returns aggregate stats matching REST
11. Verify auditability: create correction via REST, query via MCP `list_data_corrections(corrected_by="api")`, verify corrected_by, created_at, reason all populated
12. AST parse all modified files: b2b_dashboard.py, b2b_churn_server.py

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

## Execution boundaries for 3 senior devs

Do not split ownership strictly by roadmap phase. The phases reuse the same backend paths, so phase-based staffing will create collisions in `b2b_churn_intelligence.py`, migrations, dashboard APIs, and shared config.

Use three fixed lanes instead.

### Senior dev 1: test and integration lead

Primary role: cross-phase test ownership and final acceptance.

Owns:

- acceptance criteria for every sprint and phase
- integration tests, regression tests, smoke tests, and release sign-off
- shared test fixtures, test data, and verification scripts
- interface contracts between data layer, APIs, MCP tools, and delivery surfaces
- final go/no-go call on phase completion

Can change:

- test files
- verification scripts
- minimal production hooks required for testability or observability

Should not own:

- feature implementation in core intelligence or delivery layers unless needed to unblock testing

### Senior dev 2: data and intelligence platform owner

Primary role: the moat layer.

Owns:

- Phases 0 through 3 implementation
- data models, migrations, ingestion, normalization, enrichment, snapshots, and change-event logic
- intelligence scoring and evidence packing
- scheduler/task wiring for intelligence generation

Primary file zones:

- `atlas_brain/autonomous/tasks/b2b_scrape_intake.py`
- `atlas_brain/services/scraping/browser.py`
- `atlas_brain/services/b2b/`
- `atlas_brain/storage/migrations/`
- `atlas_brain/config.py` for B2B scrape/intelligence settings
- `atlas_brain/services/b2b/product_matching.py`

Must not own:

- dashboard/UI presentation logic
- webhook subscription UX
- CRM/public delivery contracts beyond the backend payload contract

### Senior dev 3: delivery, action, and analyst controls owner

Primary role: thin surfaces on top of the moat.

Owns:

- Phases 4 through 6 implementation, except core scoring internals owned by senior dev 2
- REST endpoints, MCP tools, webhook delivery, CRM ingestion surfaces, PDF export, and analyst-control flows
- UI updates needed to expose existing backend capabilities without embedding business logic

Primary file zones:

- `atlas_brain/api/b2b_tenant_dashboard.py`
- `atlas_brain/api/b2b_crm_events.py`
- `atlas_brain/api/b2b_scrape.py`
- `atlas_brain/alerts/delivery.py`
- `atlas_brain/services/b2b/pdf_renderer.py`
- MCP-facing service code
- frontend B2B pages that consume backend APIs

Must not own:

- ingestion internals
- canonicalization logic
- snapshot/change-event internals
- ranking/scoring changes without approval from senior dev 2 and test sign-off from senior dev 1

### Working rules to prevent conflicts

1. One owner per file path at a time. If a change crosses ownership lines, the non-owner opens the contract first and the owner lands the shared-file change.
2. Schema-first rule. Migration and table-shape changes start with senior dev 2, then senior dev 3 consumes them, then senior dev 1 validates them.
3. API-contract rule. Request/response shape changes are proposed by senior dev 3, reviewed by senior dev 2 for data correctness, and locked by senior dev 1 in tests.
4. Test-gate rule. No phase is called complete until senior dev 1 signs off against the acceptance checklist in this roadmap.
5. No UI-only logic. If a rule affects scoring, filtering, evidence, suppression, or correction behavior, it belongs in backend code, not the UI.
6. Branch discipline. Each senior dev works on a separate branch. Shared-file work merges in this order: data owner, delivery owner, then test lead verification.
7. Handoff discipline. Every completed sprint handoff must include changed files, migration IDs, new env vars, endpoints/tools added, and exact test steps.

### Default execution order per sprint

1. Senior dev 1 writes or updates the acceptance checklist and regression targets.
2. Senior dev 2 lands backend data/intelligence changes and migrations.
3. Senior dev 3 lands API, MCP, webhook, PDF, and UI surface changes against the stabilized backend contract.
4. Senior dev 1 runs integration/regression validation and either signs off or sends defects back to the owning lane.

---

## Sprint 1 execution checklist

Sprint 1 maps to Phase 0: current-state hardening.

Goal for this sprint:

- eliminate source-name drift
- make vendor normalization explicit and reusable
- guarantee provenance on derived intelligence rows
- expose measurable source-health output for validation

### Scope in

- source normalization helper consolidation
- vendor registry and canonicalization wiring verification
- provenance field enforcement in intelligence aggregation paths
- source-health metrics API and validation path

### Scope out

- new delivery surfaces
- major UI redesign
- historical snapshots
- CRM outcome ingestion
- score calibration

### Senior dev 2 deliverables

Must deliver:

- one canonical source parsing path used across scrape intake, enrichment, reporting, and any downstream filters
- vendor normalization path documented at all ingest and synthesis touchpoints
- any migration or schema patch needed to enforce provenance fields or vendor registry integrity
- backend source-health aggregation that exposes success rate, block rate, and usable yield by source
- a short implementation note listing exact files changed, migration IDs, and any env/config changes

Must attach for handoff:

- before/after examples of mixed source labels resolved to canonical values
- one example vendor alias path showing raw input -> canonical output
- one SQL snippet or API example proving intelligence rows retain evidence provenance

### Senior dev 3 deliverables

Must deliver:

- API or MCP exposure for source-health inspection if backend contract changed
- any thin dashboard wiring required to surface source-health without adding business logic in the client
- request/response examples for any endpoint or tool added or changed
- a short implementation note listing changed endpoints, payload shapes, auth expectations, and UI files touched

Must not do in Sprint 1:

- reimplement canonicalization rules in API or UI code
- add client-side derivation for source-health calculations

### Senior dev 1 validation checklist

Validate source normalization:

1. Run or inspect a representative ingest path and verify downstream records do not contain mixed labels for the same source.
2. Confirm one shared parsing/helper path is being used rather than duplicated source-name maps.
3. Verify allowlist/filter inputs normalize aliases correctly.

Validate vendor normalization:

1. Confirm vendor aliases resolve to one canonical vendor at ingest touchpoints and synthesis touchpoints.
2. Verify registry-backed resolution is used, not ad hoc string cleanup.
3. Check that no new duplicate canonical vendors were introduced by the sprint changes.

Validate provenance:

1. Verify derived intelligence rows can be traced to raw evidence with source, source_url or equivalent, review_id, and timestamps.
2. Verify at least one report/query path returns or preserves provenance fields rather than dropping them.
3. Spot check that null or placeholder provenance is not silently accepted where evidence is expected.

Validate source health:

1. Verify source-health output exists and is queryable through the agreed API or MCP surface.
2. Confirm metrics include success rate, block rate, and review or usable-record yield.
3. Confirm output is aggregated by canonical source name, not raw aliases.

Regression checks:

1. Existing intelligence generation still completes.
2. Existing dashboard/report reads still work for unchanged payload consumers.
3. No UI-only logic was added for normalization, provenance, or health scoring.

### Required handoff packet from each implementation owner

Each owner must provide:

- changed files
- migration IDs applied
- new env vars or config flags
- endpoints or MCP tools added or changed
- exact commands or requests needed to verify the work
- known risks or deferred gaps

### Sprint 1 exit criteria

Sprint 1 is accepted only if all of the following are true:

- downstream intelligence no longer shows mixed source labels for the same source family
- canonical vendor resolution is consistently applied at the declared touchpoints
- derived intelligence rows are evidence-traceable
- source-level health can be measured through a reusable backend surface
- no ownership-boundary violations were required to ship the sprint

### Definition of fail for Sprint 1

Sprint 1 is not complete if any of the following occur:

- duplicate normalization logic exists in multiple layers
- provenance exists in schema only but is not actually populated in live paths
- source-health metrics are visible only in UI code or manual SQL with no reusable contract
- implementation changed shared files without the owning lane landing the final version

---

## Sprint 2 execution checklist

Sprint 2 maps to Phase 1: managed intelligence substrate.

Goal for this sprint:

- make collection behavior explicit, schedulable, and measurable
- persist scrape telemetry needed for operating decisions
- version parsers so extraction improvements are auditable
- expose source capability and operational state through backend contracts

### Scope in

- source orchestration behavior and policy wiring
- source capability profile registry
- scrape telemetry persistence
- parser version persistence and verification

### Scope out

- canonical intelligence graph expansion
- historical snapshots and change events
- delivery-channel expansion beyond operational inspection surfaces
- analyst correction workflows

### Senior dev 2 deliverables

Must deliver:

- one orchestration path that makes priority, concurrency, retry, cooldown, and fallback behavior explicit in backend code
- capability profiles for supported sources with stable fields for access pattern, anti-bot class, proxy requirements, and expected freshness
- persisted scrape telemetry for attempts and outcomes, including status, duration, yield, and relevant anti-bot or proxy metadata available in the current architecture
- parser version persisted on extraction runs so future reprocessing decisions are possible
- a short implementation note listing exact files changed, migration IDs, scheduler/task impacts, and config changes

Must attach for handoff:

- one example of orchestration policy for an API source and one for a protected web source
- one sample telemetry row or query showing populated operational fields
- one example scrape result proving parser version is stored end to end

### Senior dev 3 deliverables

Must deliver:

- API or MCP exposure for source capabilities and scrape-health inspection if contract changes require it
- any thin dashboard wiring needed to inspect source capabilities, telemetry, or source health without moving logic client-side
- request/response examples for any endpoint or tool added or changed
- a short implementation note listing changed endpoints, payload shapes, auth expectations, and UI files touched

Must not do in Sprint 2:

- encode retry, cooldown, or fallback policy in API or UI layers
- compute source capability or telemetry summaries exclusively in the client

### Senior dev 1 validation checklist

Validate orchestration:

1. Confirm collection code has one explicit orchestration path rather than source-specific policy duplicated across call sites.
2. Verify priority, concurrency, retry, cooldown, and fallback behavior are inspectable in code and align with the sprint contract.
3. Confirm protected-source handling does not regress simpler API-source execution.

Validate capability profiles:

1. Verify each supported source exposes a stable capability profile through backend code.
2. Confirm capability fields are reusable by APIs or tools without UI-only transformation.
3. Spot check that source names in capability output use canonical labels.

Validate scrape telemetry:

1. Verify scrape attempts and outcomes persist enough data to answer which sources and routes are effective.
2. Confirm telemetry includes source, status, duration, yield, and relevant proxy or anti-bot context available from the implementation.
3. Confirm telemetry is queryable through a reusable backend path, not only ad hoc SQL.

Validate parser versioning:

1. Verify parser version is stored on scrape runs and tied to the relevant extraction path.
2. Confirm version values are populated, not left null or hard-coded placeholders.
3. Confirm existing scrape flows still succeed with version persistence enabled.

Regression checks:

1. Existing ingest paths still complete for at least one API source and one web source.
2. Source-health inspection from Sprint 1 still works after telemetry changes.
3. No delivery layer duplicated orchestration or telemetry logic.

### Required handoff packet from each implementation owner

Each owner must provide:

- changed files
- migration IDs applied
- new env vars or config flags
- endpoints or MCP tools added or changed
- exact commands or requests needed to verify the work
- known risks or deferred gaps

### Sprint 2 exit criteria

Sprint 2 is accepted only if all of the following are true:

- collection policy is explicit and centrally owned in backend code
- source capabilities are queryable through a stable backend contract
- scrape telemetry is persisted with enough operational detail to compare source performance
- parser version is stored on extraction runs
- no ownership-boundary violations were required to ship the sprint

### Definition of fail for Sprint 2

Sprint 2 is not complete if any of the following occur:

- orchestration policy is still fragmented across multiple source-specific paths without a clear owner
- telemetry exists partially but does not capture enough data to compare source effectiveness
- parser version fields exist in schema but are not populated on live scrape paths
- capabilities or telemetry are only visible through UI code or manual inspection with no reusable contract
- implementation changed shared files without the owning lane landing the final version

---

## Sprint 5 execution checklist

Sprint 5 maps to Phase 4: action feedback loop.

Goal for this sprint:

- make campaign and CRM outcomes queryable through stable backend contracts
- connect intelligence outputs to downstream action results
- expose action-performance surfaces without moving logic into the UI

### Scope in

- campaign outcome tracking surfaces
- CRM event ingestion surfaces and normalization
- action-performance reads and calibration exposure

### Scope out

- score-model internals beyond backend hooks owned by senior dev 2
- webhook subscription delivery
- PDF export
- analyst correction workflows

### Senior dev 1 jobs

1. Define the acceptance pack for outcome tracking, CRM ingest, auth, dedup, and attribution reads.
2. Freeze the regression pack for existing campaign, dashboard, and intelligence flows affected by CRM ingestion.
3. Validate end-to-end that outcome events can be ingested, processed, queried, and traced back to source actions.

### Senior dev 2 jobs

1. Own backend outcome persistence and attribution correctness for campaign and intelligence-linked events.
2. Own calibration data-model hooks and backend score-adjustment interfaces used by delivery surfaces.
3. Own processing-task behavior for event ranking, matching, and skip rules where business logic lives.

### Senior dev 3 jobs

1. Own REST and MCP surfaces for campaign outcomes, CRM event ingestion, filtered event reads, and calibration-weight reads.
2. Own HubSpot, Salesforce, and Pipedrive ingress surface contracts, auth behavior, validation, and dedup exposure.
3. Own thin dashboard/admin views for outcome history, ingest status, and action-performance reporting.

### Sprint 5 exit criteria

- campaign outcomes are queryable outside the UI
- CRM events can be ingested and normalized through stable backend contracts
- dedup, auth, and validation are enforced at the API layer
- action-performance surfaces expose backend truth without client-side business logic

### Definition of fail for Sprint 5

- CRM ingest works only for one provider while contract shape drifts across others
- attribution or calibration reads depend on UI-only logic
- outcome events cannot be traced through processing and final reads
- implementation changed shared files without the owning lane landing the final version

---

## Sprint 6 execution checklist

Sprint 6 maps to Phase 5: thin delivery surfaces.

Goal for this sprint:

- make delivery channels reusable outside the dashboard
- ensure API, webhook, and export surfaces are first-class
- keep the UI thin and replaceable

### Scope in

- webhook subscription and delivery surfaces
- PDF intelligence export surfaces
- API-first delivery contracts and thin UI exposure

### Scope out

- CRM ingestion internals
- canonical scoring logic
- analyst correction execution

### Senior dev 1 jobs

1. Define acceptance checks for webhook subscription lifecycle, delivery logging, HMAC signing, PDF export, and access control.
2. Run regression on existing report generation and alerting paths affected by new delivery surfaces.
3. Validate that customers can retrieve value through API, webhook, and PDF without requiring UI-derived logic.

### Senior dev 2 jobs

1. Own backend event payload correctness for change events, churn alerts, and report export data assembly.
2. Own any shared backend hooks required for webhook dispatch or PDF payload generation where business logic originates.
3. Protect core intelligence logic from drifting into delivery-layer adapters.

### Senior dev 3 jobs

1. Own webhook subscription APIs, test-send flows, delivery-history reads, and signature behavior.
2. Own PDF export endpoints, MCP export tools, response metadata, and delivery-surface auth rules.
3. Own thin dashboard/admin views for subscriptions, delivery status, and export access without embedding business logic.

### Sprint 6 exit criteria

- webhook delivery is manageable through stable APIs and observable through delivery logs
- PDF exports are retrievable through API and MCP surfaces with correct access rules
- important intelligence value is consumable without logging into the dashboard
- no business logic required for delivery lives only in UI code

### Definition of fail for Sprint 6

- webhook behavior is not reproducible through stable contracts
- PDF export works only from one surface while other consumers lack parity
- delivery payload rules are duplicated across adapters or client code
- implementation changed shared files without the owning lane landing the final version

---

## Sprint 7 execution checklist

Sprint 7 maps to Phase 6: analyst controls.

Goal for this sprint:

- make analyst corrections durable, auditable, and reusable
- expose correction workflows through stable backend surfaces
- ensure manual quality improvements compound over time

### Scope in

- correction-management APIs and MCP tools
- override, merge, suppression, and impact-inspection surfaces
- thin admin views for analyst operations and audit history

### Scope out

- deep canonicalization internals beyond hooks owned by senior dev 2
- core ingestion orchestration
- new external delivery channels

### Senior dev 1 jobs

1. Define acceptance checks for correction creation, impact inspection, revert behavior, merge flows, override reads, and source suppression.
2. Freeze regression checks for reads affected by corrections so behavior changes are intentional and traceable.
3. Validate auditability: who changed what, why, when, and what records were affected.

### Senior dev 2 jobs

1. Own backend execution logic for correction application, merge semantics, suppression semantics, and read-path correctness where domain rules live.
2. Own data-model integrity for corrections, affected-record accounting, and correction-to-pipeline feedback hooks.
3. Own protection against duplicated correction logic appearing in surface layers.

### Senior dev 3 jobs

1. Own correction-management REST and MCP surfaces, including create, list, inspect-impact, and revert flows.
2. Own thin analyst/admin UI for reviewing corrections, applying actions, and inspecting audit history.
3. Own payload contracts and auth behavior for override, merge, and suppression operations without reimplementing backend rules.

### Sprint 7 exit criteria

- analyst corrections are durable, queryable, auditable, and reversible through stable backend contracts
- corrected values appear on the intended backend read paths
- repeated cleanup no longer requires direct DB intervention
- UI remains a thin operator surface over backend correction logic

### Definition of fail for Sprint 7

- correction effects are visible only in UI code and not backend reads
- merges, overrides, or suppressions are not auditable end to end
- surface layers reimplement correction logic owned by backend services
- implementation changed shared files without the owning lane landing the final version

---

## Active job assignments

Phases 1 through 6 are complete. Active work now shifts to Phase 7, full-system verification, deferred-gap triage, and launch-readiness.

### Senior dev 1: test and integration lead

Current jobs:

1. Run final end-to-end validation across completed Phases 1 through 6.
2. Build the Phase 7 acceptance pack for messaging consistency, packaging consistency, and proof that the moat is exposed outside the UI.
3. Produce the completion sign-off report covering passed checks, defects, deferred items, and release risk.

Immediate deliverables:

- one final regression matrix covering ingestion, intelligence, history, CRM, webhook, PDF, MCP, and analyst-control flows
- one acceptance checklist for Phase 7 wording and packaging consistency across docs, APIs, UI labels, and exports
- one defect log classifying issues as blocker, post-launch, or deferred-gap
- one release sign-off template for declaring Phases 1 through 7 complete

Done when:

- end-to-end validation is documented and runnable
- all remaining defects are classified and assigned
- a final sign-off recommendation exists with explicit risks

### Senior dev 2: data and intelligence platform owner

Current jobs:

1. Audit completed Phases 1 through 6 against the roadmap's stated deliverables and deferred gaps.
2. Triage any remaining backend correctness gaps, especially where completed phases left explicit follow-up items.
3. Produce the technical source-of-truth summary for the moat: what data assets, models, jobs, and APIs now exist.
4. Support Phase 7 by defining the exact technical language that should replace weaker product wording.

Immediate deliverables:

- one backend completion audit mapped to roadmap claims and actual implementation
- one deferred-gap list separating true defects from intentional backlog items
- one moat-asset inventory covering ingestion, normalization, historical memory, action feedback, delivery, and analyst controls
- one terminology note defining approved technical phrases for product and operational use

Done when:

- roadmap claims are technically defensible
- backend deferred gaps are clearly separated from shipped functionality
- the architecture can be described consistently without UI-first language

### Senior dev 3: delivery, action, and analyst controls owner

Current jobs:

1. Execute Phase 7 across customer-visible and operator-visible surfaces.
2. Replace weak or UI-first language in docs, API descriptions, labels, report names, and delivery packaging.
3. Standardize packaging around feeds, watchlists, exports, alerts, and intelligence products rather than generic dashboards.
4. Audit thin-client compliance so no newly completed Phase 4 through 6 surface hides moat logic in the UI.

Immediate deliverables:

- one copy and naming pass across UI labels, endpoint descriptions, tool descriptions, and exports
- one packaging matrix mapping each intelligence product to its delivery surfaces
- one thin-client audit showing which surfaces are API-first and which still need cleanup
- one change list for any user-facing wording updated to match Phase 7

Done when:

- user-facing language consistently reflects the moat architecture
- delivery packaging emphasizes feeds, exports, alerts, and intelligence products
- UI wording no longer implies the dashboard is the core product

### Coordination jobs for all three seniors

All three must do the following before final close-out:

1. Reconcile completed work against the roadmap and mark any remaining gaps explicitly.
2. Use the required handoff packet: changed files, migrations, env vars, endpoints, verification steps, risks.
3. Escalate any roadmap claim that is not technically defensible.
4. Treat completion status, release messaging, and moat claims as controlled surfaces.

### Recommended start order now

1. Senior dev 2 publishes the backend completion audit and deferred-gap inventory.
2. Senior dev 3 updates product language and delivery packaging based on the approved moat terminology.
3. Senior dev 1 runs full-system regression and Phase 7 acceptance against the completed implementation.
4. All three review blockers, post-launch items, and the final release claim before sign-off.

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

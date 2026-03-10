# Consumer Intelligence Platform Roadmap

## Purpose

Port applicable B2B intelligence infrastructure to the consumer product review pipeline.
This is a **separate tracking doc** from `intelligence_platform_roadmap.md` (which is B2B-only).

Reference: The B2B roadmap has 6 phases. Not all apply to consumer (batch import, no campaigns).
This doc tracks only what's applicable and its consumer-specific implementation.

---

## Current State (March 2026)

### Already Ported
| Feature | B2B Source | Consumer Implementation | Migration |
|---------|-----------|------------------------|-----------|
| Daily brand snapshots | `b2b_vendor_snapshots` | `brand_intelligence_snapshots` | 121 |
| Change event detection | `b2b_change_events` | `product_change_events` | 121 |
| Corrections table | `data_corrections` | Extended CHECK constraint | 122 |
| Suppress-aware queries | `suppress_predicate()` | Applied to review search + export | -- |
| Field override reads | `apply_field_overrides()` | Applied to single review GET | -- |
| Corrections CRUD REST | 5 endpoints | 5 endpoints on consumer_dashboard | -- |
| Corrections MCP tools | 4 tools | 3 tools on intelligence_server | -- |
| History REST endpoints | vendor-history, change-events | brand-history, change-events, summary | -- |
| History MCP tools | get_vendor_history, list_change_events | get_brand_history, list_product_change_events | -- |
| CSV/JSON export | signals, reviews, high-intent | reviews, brands, pain-points | -- |
| ntfy notifications | Wired in runner.py | Wired in competitive_intelligence.py | -- |

### Not Applicable (Skipped)
| B2B Feature | Reason |
|-------------|--------|
| Phase 1: Scrape orchestration, telemetry, parser versioning | Consumer is batch import, not scraped |
| Phase 4: Campaign outcomes, CRM events | Consumer has no campaigns |
| Phase 6: Source suppression | Single source (Amazon) |

---

## Phase 0: Brand Identity & Normalization -- DONE

**B2B precedent**: `b2b_vendor_registry` table (migration 095) with canonical_name + aliases array.
`resolve_vendor_name()` at ingest. Fuzzy matching via pg_trgm (migration 114) + Python difflib fallback.

**Why it matters**: Brand names are raw strings from product_metadata. "Apple" vs "Apple Inc." vs
"apple" fragment snapshots, change events, and reports. Without a canonical registry, all
downstream aggregation is unreliable.

| Item | Status | What's There |
|------|--------|-------------|
| Brand registry table | **DONE** | `consumer_brand_registry` table (migration 123): canonical_name UNIQUE, aliases JSONB, metadata JSONB, GIN + GiST indexes. 3,850 brands seeded from product_metadata. |
| Brand resolution function | **DONE** | `brand_registry.py`: `resolve_brand_name()` (async, DB-backed) + `resolve_brand_name_cached()` (sync, cache-only). 5-min TTL cache with double-check locking. Bootstrap aliases for cold start. |
| Fuzzy brand matching | **DONE** | pg_trgm GiST index on canonical_name. `fuzzy_search_brands()` with configurable similarity threshold. Python difflib fallback (SequenceMatcher >= 0.85). |
| Ingest-time normalization | **DONE** | `_fetch_brand_health()` warms cache via `_ensure_brand_cache()`, resolves all brand names via `resolve_brand_name_cached()` before snapshot/report persistence. |
| Brand provenance fields | **DONE** | `source_review_count` INTEGER + `source_distribution` JSONB on `brand_intelligence` (migration 123). Populated in `_upsert_brand_intelligence()`. |
| `load_known_brands()` integration | **DONE** | `comparisons.py:load_known_brands()` merges registry canonical names + aliases into known brands dict. All downstream callers (competitive flows, brand detail, dashboard) automatically benefit. |
| REST endpoints | **DONE** | `GET /fuzzy-brand-search`, `GET /brand-registry`, `POST /brand-registry`, `POST /brand-registry/{brand}/aliases` |
| MCP tools | **DONE** | `fuzzy_brand_search`, `add_brand_to_registry`, `add_brand_alias`, `list_brand_registry` (28 tools total) |

---

## Phase 2: Canonical Entities & Confidence Scoring -- DONE

**B2B precedent**: `b2b_displacement_edges` (append-only time-series), `b2b_vendor_pain_points`
(per vendor-category with confidence), `b2b_vendor_use_cases`, `b2b_vendor_buyer_profiles`.
3-signal confidence scoring: mention weight (log-scaled) + source diversity + verified source proportion.

**Why it matters**: Competitive flows and pain points are trapped in JSONB on reports.
Can't query "which brands lost share to Brand X over time" or filter by confidence.
Pain scores have no evidence backing -- can't distinguish strong signals from noise.

| Item | Status | What's There |
|------|--------|-------------|
| Product displacement edges | **DONE** | `product_displacement_edges` table (migration 124): from_brand, to_brand, direction, mention_count, signal_strength, avg_rating, category_distribution JSONB, sample_review_ids UUID[], confidence_score, computed_date. UNIQUE per (from_brand, to_brand, direction, computed_date). 5 indexes. |
| Confidence scoring function | **DONE** | `_compute_consumer_confidence()`: 3-signal (mention weight log2/50, enrichment depth ratio, severity entropy consistency). Adapted from B2B -- replaces source diversity with enrichment depth, verified sources with severity consistency. |
| Confidence on brand_intelligence | **DONE** | `confidence_score NUMERIC(3,2)` added (migration 124). Computed in `_upsert_brand_intelligence()` from total_reviews + severity_distribution. |
| Confidence on pain_points | **DONE** | `confidence_score NUMERIC(3,2)` added (migration 124). Column ready for `complaint_analysis.py` to populate. |
| Displacement edge persistence | **DONE** | `_persist_displacement_edges()` in competitive_intelligence.py. Extracts flows from deep_extraction product_comparisons JSONB, normalizes brands through registry, aggregates per (from, to, direction), filters noise (< 2 mentions), computes confidence + signal strength. Wired into daily `run()`. |
| min_confidence filter | **DONE** | Added to `GET /export/brands` and `GET /export/pain-points`. Displacement edge endpoints have min_confidence natively. |
| Displacement edge REST | **DONE** | `GET /displacement-edges` (filtered, sorted by mention_count + confidence) + `GET /displacement-history` (time-series for brand pair). |
| Displacement edge MCP | **DONE** | `list_product_displacement_edges` + `get_product_displacement_history` (30 tools total). |

---

## Phase 3: Cross-Brand Correlation -- DONE

**B2B precedent**: `GET /concurrent-events` (dates where 3+ vendors had same event type).
`GET /vendor-correlation` (pairwise Pearson r on snapshot time-series with displacement context).
Automatic `concurrent_shift` event detection.

| Item | Status | What's There |
|------|--------|-------------|
| Concurrent event detection | **DONE** | `GET /consumer/dashboard/concurrent-events`: dates where N+ brands had same event type. Filters by event_type, min_brands, days. Aggregates avg/min/max delta. |
| Pairwise brand correlation | **DONE** | `GET /consumer/dashboard/brand-correlation?brand_a=X&brand_b=Y&metric=avg_pain_score`: aligned `brand_intelligence_snapshots` time-series + Pearson r. 8 metrics supported: health_score, avg_pain_score, avg_rating, total_reviews, repurchase_yes, safety_count, complaint_count, competitive_flow_count. Includes recent displacement edges between the pair. |
| Market-level signal detection | **DONE** | `_detect_concurrent_shifts()` in competitive_intelligence.py: runs daily after change events, detects 3+ brands with same event type, creates `concurrent_shift` event with brand=`__market__` pseudo-brand + metadata (original_event_type, brand_count, brands). |
| MCP tools | **DONE** | `list_concurrent_events` (filterable, same query as REST), `get_brand_correlation` (inline Pearson r, displacement context). 32 tools total. |

---

## Phase 5: Delivery Surfaces -- NOT STARTED

**B2B precedent**: Webhook outbound delivery (HMAC-SHA256 signed), PDF export (fpdf2),
Slack/Teams notifications (Block Kit, Adaptive Cards), email digest (Resend).

**Already done**: CSV/JSON export (3 endpoints), ntfy notifications.

| Item | Status | What's Needed |
|------|--------|---------------|
| Webhook delivery | **MISSING** | Reuse `b2b_webhook_subscriptions` infrastructure. Add consumer event types to webhook dispatch. |
| PDF brand report export | **MISSING** | Extend `pdf_renderer.py` or create consumer-specific renderer. MCP tool: `export_brand_report_pdf`. |
| Slack/Teams notifications | **MISSING** | Reuse webhook_dispatcher channel formatters. Wire consumer change events into dispatch. |
| Email digest | **MISSING** | Weekly brand health summary email. Reuse Resend provider. |

---

## Phase 6: Brand Merge & Advanced Controls -- DONE

**B2B precedent**: `merge_vendor` correction type renames across 17 tables in transaction,
adds old name as alias, invalidates cache.

| Item | Status | What's There |
|------|--------|-------------|
| Brand merge service | **DONE** | `brand_merge.py`: `execute_brand_merge()` renames across brand_intelligence (UNIQUE brand+source), brand_intelligence_snapshots (UNIQUE brand+snapshot_date), product_change_events, product_metadata, product_displacement_edges (UNIQUE from_brand+to_brand+direction+computed_date, both columns). Two-phase conflict handling (DELETE conflicts, then UPDATE). Refreshes mv_brand_summary. Adds old name as alias in consumer_brand_registry. Invalidates resolution cache. |
| Merge REST endpoint | **DONE** | `POST /consumer/dashboard/corrections` with `correction_type=merge_brand`, old_value=source, new_value=target. Stores merge result in correction metadata + affected_count. |
| Merge MCP tool | **DONE** | `create_consumer_correction` extended with `merge_brand` support. Same execution path, returns merge stats in response. |

---

## Implementation Order

```
Phase 0  (brand identity)     -- foundational, must be first
Phase 2  (canonical entities) -- depends on Phase 0 for brand resolution
Phase 3  (cross-brand)        -- depends on Phase 3 snapshots (done) + Phase 2 entities
Phase 6  (brand merge)        -- depends on Phase 0 brand registry
Phase 5  (delivery surfaces)  -- independent, can be done anytime after Phase 0
```

---

## Overall Score

| Phase | Score | Summary |
|-------|-------|---------|
| Phase 0 | **100%** | Brand registry (3,850 seeded), resolution with cache, fuzzy matching, provenance, REST + MCP |
| Phase 2 | **100%** | Displacement edges, confidence scoring (3-signal), min_confidence filters, REST + MCP |
| Phase 3 | **100%** | Concurrent event detection, pairwise Pearson r correlation, market-level concurrent_shift events, REST + MCP |
| Phase 5 | **30%** | CSV export + ntfy done. Missing webhooks, PDF, Slack/Teams, email. |
| Phase 6 | **100%** | Brand merge across 6 tables (conflict-aware), mv refresh, alias + cache invalidation, REST + MCP |

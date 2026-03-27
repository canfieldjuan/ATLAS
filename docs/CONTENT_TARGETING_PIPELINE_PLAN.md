# Content Targeting Pipeline — Implementation Plan

**Created:** 2026-03-21
**Status:** Phase 0 scoped, awaiting approval
**Depends on:** Reasoning pool audit complete (17 gaps fixed across 6 pools, 8 files)

---

## Context

The B2B churn pipeline has 6 reasoning pools that produce confidence-scored, cross-correlated intelligence. Battle cards, challenger briefs, vendor briefings, and churn reports all consume this reasoned data. Blog post generation does NOT — it still reads raw aggregated tables.

Blog generation is currently **paused** (`.env.local: ATLAS_B2B_CHURN_BLOG_POST_ENABLED=false`).

---

## Phase 0: Rewire blog generation to reasoning pools

**Goal:** Blog posts built from the same reasoned intelligence as battle cards and briefs, not raw aggregates.

**Problem:** `_gather_data()` in `b2b_blog_post_generation.py` loads raw `b2b_churn_signals` + `b2b_reviews` + `b2b_product_profiles`. The 6 reasoning pools (evidence vault partially used, 5 others absent) are never loaded. The LLM writing a "Salesforce vs HubSpot" showdown gets pain counts and profile bullets but NOT displacement dynamics, battle conclusions, per-role churn, market regime, or causal narratives.

**Fix:**

1. New function `_load_pool_layers_for_blog(pool, topic_type, topic_ctx, data)` in `b2b_blog_post_generation.py`
   - Calls `fetch_all_pool_layers(pool, as_of=today, analysis_window_days=window_days)` (from `_b2b_shared.py:3768`)
   - Extracts vendor-specific layers based on `topic_ctx` vendor names
   - Injects into `data` dict as additive keys (`data["displacement"]`, `data["segment"]`, etc.)
   - Optionally loads reasoning synthesis via `load_synthesis_view()` for causal narratives

2. Call in `run()` between `_gather_data()` (line 159) and `_check_data_sufficiency()` (line 161)

3. Update blueprint functions to inject pool data into `key_stats` and `data_summary`:
   - `_blueprint_vendor_showdown` (line 2238): displacement edge, battle conclusion, category regime
   - `_blueprint_vendor_alternative` (line 2118): displacement velocity, temporal triggers
   - `_blueprint_churn_report`: all 6 pools for the vendor
   - `_blueprint_migration_guide`: displacement dynamics both directions
   - `_blueprint_pricing_reality_check`: evidence vault correlations, segment budget pressure
   - `_blueprint_switching_story`: displacement evidence breakdown, account intelligence
   - `_blueprint_vendor_deep_dive`: full vault + segment + temporal
   - `_blueprint_market_landscape`: category dynamics, displacement flows
   - `_blueprint_pain_point_roundup`: evidence vault correlations, per-role pain
   - `_blueprint_best_fit_guide`: segment intelligence, category dynamics

4. Update blog skill prompt (`b2b_blog_post_generation.md`) to reference new reasoning fields

**Files modified:**
- `b2b_blog_post_generation.py` — new function + call site + 10 blueprint functions
- `skills/digest/b2b_blog_post_generation.md` — reference new data fields

**Files NOT modified:**
- `_b2b_shared.py` — `fetch_all_pool_layers()` already exists
- `_b2b_synthesis_reader.py` — `load_synthesis_view()` already exists
- No migrations — pool data injected into existing `data_context` JSONB

**Breaking changes:** None. All pool data is additive via `.get()` with fallbacks.

**Data available per topic type:**

| Topic Type | New Intelligence Added |
|------------|----------------------|
| `vendor_showdown` | Displacement dynamics (A->B edge, switch reasons, velocity), battle conclusion (winner/loser/confidence/durability), category dynamics (market regime), segment (affected roles, budget) |
| `vendor_alternative` | Displacement velocity, temporal triggers (deadlines, spikes), account intelligence (high-intent companies) |
| `churn_report` | All 6 pools — full intelligence picture |
| `migration_guide` | Displacement dynamics both directions, temporal (contract end signals) |
| `pricing_reality_check` | Evidence vault (correlated pricing weakness, confidence scored), segment (budget pressure, per-role churn) |
| `switching_story` | Displacement (switch reasons, evidence breakdown), account intelligence |
| `vendor_deep_dive` | Evidence vault (full weakness/strength with correlations), segment, temporal |
| `market_landscape` | Category dynamics (regime, council conclusion), displacement flows across category |
| `pain_point_roundup` | Evidence vault (cross-weakness correlations), segment (per-role pain) |
| `best_fit_guide` | Segment (company size, use cases), category dynamics |

**Status:** [x] Complete. All 4 steps done.

**Completed:**
- `_load_pool_layers_for_blog()` — loads all 6 pools + reasoning synthesis, extracts vendor-specific shortcuts
- Wired into both `run()` call sites (main path line 160, regenerate path line 304)
- All 10 blueprint functions updated with pool data injection:
  - `vendor_showdown`: displacement edge, battle conclusion, switch reasons, segment roles, category regime, synthesis wedge
  - `vendor_alternative`: displacement targets, temporal triggers, keyword spikes, segment budget, causal narrative
  - `churn_report`: displacement targets, segment roles + budget, temporal renewal/sentiment, causal narrative
  - `migration_guide`: displacement driver + velocity, causal trigger
  - `vendor_deep_dive`: segment roles + churn rate, sentiment trajectory, synthesis wedge
  - `market_landscape`: category regime, council conclusion + winner
  - `pricing_reality_check`: segment budget pressure + price increase rate, top churning role
  - `switching_story`: displacement destination + driver + switch reasons
  - `pain_point_roundup`: category regime, outlier vendors
  - `best_fit_guide`: category winner + conclusion

---

## Phase 1: Pain-aware blog matching

**Goal:** Campaign emails link to blog posts that match the target's pain category, not just the vendor name.

**Problem:** `fetch_relevant_blog_posts()` in `_blog_matching.py` scores by vendor name (+3) and category (+2) only. A pricing-focused campaign could link to a feature comparison blog.

**Fix:**
1. Add `pain_categories: list[str] | None = None` param to `fetch_relevant_blog_posts()` (line 45)
2. Query `blog_posts.data_context` JSONB for pain category overlap
3. Score: +4 for pain match
4. Campaign generation passes top pain from context to blog fetch (lines 704, 1143, 1536 in `b2b_campaign_generation.py`)

**Files modified:**
- `_blog_matching.py` — signature + scoring algorithm
- `b2b_campaign_generation.py` — 3 blog fetch call sites

**Breaking changes:** None. New param is optional with default `None`.

**Status:** [x] Complete. Implemented together with Phase 2.

---

## Phase 2: Alternative-aware matching

**Goal:** Campaign targeting someone evaluating HubSpot links to the "X vs HubSpot" showdown post.

**Problem:** Matching searches by incumbent vendor only. Showdown posts have `vendor_b` in `data_context` but matching doesn't query it.

**Fix:**
1. Add `alternative_vendors: list[str] | None = None` param to `fetch_relevant_blog_posts()`
2. Query `data_context->>'vendor_b'` and `data_context->'topic_ctx'->>'vendor_b'` for match
3. Score: +5 for alternative match (highest signal)
4. Challenger campaigns pass incumbent names; affiliate campaigns pass `churning_from`

**Files modified:**
- `_blog_matching.py` — signature + scoring
- `b2b_campaign_generation.py` — 3 blog fetch call sites

**Breaking changes:** None. New param is optional.

**Status:** [x] Complete. Implemented together with Phase 1.

---

## Phase 3: Campaign-gap-aware topic scoring

**Goal:** Blog topic selection prioritizes topics that fill content gaps — if campaigns about Salesforce pricing exist but no pricing blog does, that topic scores higher.

**Problem:** Topics scored by signal strength (urgency * reviews) not by content demand.

**Fix:**
1. Before topic scoring, query `b2b_campaigns` (last 14 days) grouped by (vendor, pain_category)
2. Cross-reference against `blog_posts` to find (vendor, pain) pairs with campaigns but no blog
3. Add `gap_bonus` multiplier to matching topic candidates

**Files modified:**
- `b2b_blog_post_generation.py` — new helper + inject into topic scoring

**Dependencies:** Blog generation must be re-enabled (`.env.local` line 144).

**Status:** [x] Complete.

---

## Execution Order

```
Phase 0 (content quality)     — blog posts use reasoned intelligence
    |
Phase 1 (pain matching)       — campaigns link to pain-relevant posts
    |
Phase 2 (alternative matching) — campaigns link to competitor-specific posts
    |
Phase 3 (gap-aware topics)    — blog generation fills campaign content gaps
```

Phase 0 is prerequisite for content quality. Phases 1-2 improve matching independent of Phase 0 but benefit from better content. Phase 3 requires blog generation to be active.

---

## Reference: Reasoning Pool Audit (2026-03-21)

17 gaps fixed across 6 pools, 8 files modified:

| Pool | Gaps Fixed |
|------|-----------|
| Evidence Vault | 8 (strength wiring, metric wiring, company signals, correlation, recency, quote matching, urgency weighting) |
| Segment Intelligence | 2 (per-role churn rates, lock-in levels) |
| Temporal Intelligence | 1 (renewal + budget cycle signals) |
| Displacement Dynamics | 1 (battle summary inversion) |
| Category Dynamics | 0 (no actionable gaps) |
| Account Intelligence | 2 (dropped fields restored: title, company_size, industry, review_id, alternatives, quotes) |

Evidence vault confidence formula now: `share + type + volume + correlation + recency + urgency` (6 components, was 3).

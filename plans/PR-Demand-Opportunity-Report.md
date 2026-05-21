# PR-Demand-Opportunity-Report

## Why this slice exists

A market/product-research effort needs "real demand" signal -- what B2B
software buyers actually complain about, wish existed, and pay for -- ranked
into product-opportunity candidates. The Atlas churn pipeline already extracts
exactly this at its enrichment stage and **the data already exists**: the live
DB holds 22,176 enriched B2B-software reviews across 13 categories (CRM has
1,959). No new scraping is required to start.

What is missing is a consumer that reads the enriched reviews through a
*demand* lens rather than the existing *churn/competitive-displacement* lens
(stages 5-8 of the pipeline: `b2b_churn_intelligence` -> ... -> blog). Those
reinterpret the data as "who is switching from vendor X to Y", which is not
market/product demand. This slice adds a small, on-demand aggregation that
branches off the pipeline **right after enrichment** and produces a structured
demand/opportunity report a human reviews and refines.

Pilot category: **CRM** (1,959 enriched reviews).

## Scope (this PR)

This PR checks in the plan only; the implementation follows in a second PR so
the report shape was reviewed first. The plan was checked in via #708; this
PR adds the implementation:

1. `scripts/demand_opportunity_report.py --category=<cat> [--json]` -- the
   on-demand, read-only report (reads the live DB via `init_database()`,
   prints markdown or JSON; no cron task, no writes), following the existing
   `scripts/audit_*` / `scripts/check_*` convention.
2. `tests/test_demand_opportunity_report.py` -- unit tests for the pure
   aggregation, the relevance filter, the baseline exclusion, and the rollups.
3. Plan-doc update recording the relevance filter (added after the first CRM
   run surfaced the contamination) and the real verification results.

### Files touched

- `scripts/demand_opportunity_report.py`
- `tests/test_demand_opportunity_report.py`
- `plans/PR-Demand-Opportunity-Report.md`

## Mechanism

**Input.** Enriched reviews for the chosen category
(`b2b_reviews WHERE enrichment_status='enriched' AND product_category=$cat`),
joined to their primary vendor via `b2b_review_vendor_mentions` (`is_primary`)
so pains can be rolled up per vendor and across the category.

**Signals used (and why the coarse field is avoided).** The single
`enrichment->>'pain_category'` is ~54% `overall_dissatisfaction` for CRM -- a
catch-all that is not actionable. The report instead mines the structured
multi-fields:
- `feature_gaps` (array) -- features users say are missing = the clearest
  unmet-need / build-this signal (present on ~23% of CRM reviews).
- `specific_complaints` (array, ~60% filled) and the per-span
  `evidence_spans` where `signal_type='complaint'` (each carries its own
  `pain_category`, `text`, `replacement_mode`) -- granular pain.
- `pain_categories` / `pain_cluster` -- finer taxonomy than the single field.
- `pricing_phrases` + `budget_signals` (price_per_seat, annual_spend_estimate,
  price_increase) -- willingness-to-pay and pricing-pain.
- `competitors_mentioned` -- which alternatives buyers cite (competitive
  demand / displacement targets).
- `positive_aspects` / `would_recommend` -- what already works (do NOT compete
  here), used to discount opportunities.
- `urgency_score` -- intensity weight.

**Opportunity ranking.** For each pain cluster / feature-gap theme:
`opportunity = frequency x mean_urgency x cross-vendor_breadth`, where
cross-vendor breadth = the share of distinct category vendors whose reviews
show the theme. A pain that is widespread, intense, AND unsolved by every
vendor in the category is a market gap (high opportunity); a pain isolated to
one weak vendor is just that vendor's problem (low opportunity). `positive_aspects`
are not folded into the numeric score (mapping a positive to the pain theme it
offsets is fuzzy); instead they are surfaced verbatim in a "what already works
well" section as a do-NOT-build counter-signal the operator weighs by hand.

**Competitor normalization.** `competitors_mentioned[].name` is tallied by a
normalized key (case-folded, trailing " CRM" stripped) so variants like
`HubSpot` / `Hubspot` / `HubSpot CRM` collapse to one entry (HubSpot: 49, not
a split 37/7/4/1); the most frequent raw spelling is shown. A small curated
`_VENDOR_ALIASES` map additionally folds a bare common-word spelling into its
canonical product name where the corpus uses both interchangeably -- e.g.
`Monday` -> `Monday.com` (PM smoke test: merges 19+17 into one 36x entry),
while a distinct variant like `Monday dev` stays separate. Extend the map as
new split-variant vendors surface per category.

**Output (the structured report).** Markdown, per category, with:
- Ranked opportunity table (theme, frequency, mean urgency, vendor breadth,
  opportunity score).
- Top feature gaps (verbatim, de-duplicated, with counts).
- What already works well (top `positive_aspects`, de-duplicated) -- a
  do-NOT-build counter-signal.
- Pricing-pain summary (price-increase mentions, spend ranges).
- Per-theme evidence: 2-3 `quotable_phrases` / complaint spans with the
  review source, so every claim is traceable (no fabricated numbers).
- Competitor map: which alternatives are cited and in what context.

**Relevance filter (added after the first CRM run).** The corpus carries the
common-word-vendor contamination (see memory: common-word-vendor-contamination):
the first CRM run surfaced video-game crafting quotes ("30 stone axes", "150
durability", "silver per day") in the pricing-pain evidence, from the "Copper"
and "Close" vendors keyword-matching unrelated content. A high-confidence
off-topic marker set (gaming/crafting/physical-material terms that never occur
in genuine SaaS reviews) excludes a contaminated review *before* any counting,
so it neither inflates themes nor surfaces as evidence; the count and affected
vendors are reported in the header for transparency. The markers are
deliberately specific to avoid false-dropping legitimate reviews (a "Silver
plan" tier or a "durable workflow" survive). The real fix is upstream
vendor-name disambiguation; this keeps the report honest until then.

**Filter limitation (known).** The markers are tuned to the contamination
*shapes* actually observed for CRM (gaming/crafting, physical materials,
audio) -- they are not a general off-topic classifier. A different off-topic
domain can still leak (e.g. a stray medical "Mirena" mention surfaced in the
CRM competitor list from a "Close"-keyword match). Per-category review of the
output, plus marker additions when a new shape appears, is expected until the
upstream disambiguation lands; the report header's dropped-count makes the
filter's reach visible.

**Human-in-the-loop.** Run on demand per category; the operator reads the
report and refines or re-scopes -- nothing is auto-published. This matches the
"requires my input, not fully automated" requirement.

## Intentional

- **Branches off after enrichment (stage 2).** Deliberately does not touch
  `b2b_churn_intelligence` and downstream -- those impose a churn/displacement
  interpretation that is wrong for demand research. The report is a parallel
  consumer of the same enriched rows.
- **Uses the existing corpus; no scraping.** For categories already covered
  (all 13), the report runs immediately. Scraping is reserved for coverage
  gaps (see Deferred).
- **Multi-field, not `pain_category`.** Ranking the coarse single field would
  return "overall_dissatisfaction" as the top "opportunity", which is useless.
  The structured fields give actionable themes.
- **Read-only, on-demand script.** No cron registration, no DB writes, no new
  tables -- it is an analysis tool, not a pipeline stage. Lives in `scripts/`.
- **Evidence-traceable.** Every ranked theme carries real review quotes/spans;
  no invented statistics (same evidence-integrity bar as the blog work).

## Deferred

- **Scraping to fill gaps.** New categories not in the corpus, or deepening
  thin ones (Email Marketing 192, Customer Messaging 83, Cybersecurity 397),
  or refreshing recency -- run `b2b_scrape_intake` + `b2b_enrichment` for the
  target, then the report. Scoped only when a gap is hit.
- **LLM unmet-need pass.** If `feature_gaps` (~23% fill) proves too sparse for
  a category, a targeted LLM extraction over the raw text of the top complaint
  clusters can densify the unmet-need signal. Add only if the field-based
  report is insufficient.
- **Other categories / cross-category view.** The pilot is CRM; generalizing
  is a flag-driven re-run, not new code.

## Verification

- `tests/test_demand_opportunity_report.py` run via pytest -> `5 passed`
  (off-topic detection incl. false-positive guard, off-topic exclusion +
  count, baseline exclusion, breadth/urgency ranking, gap/competitor rollup).
- `scripts/demand_opportunity_report.py --category=CRM` against the live DB ->
  1,954 reviews / 8 vendors (5 off-topic dropped: Copper:4, Close:1). Ranked
  themes match known CRM pains: pricing (365 reviews, urgency 4.35, 8/8
  vendors, top opportunity), support, ux, then features/integration/onboarding;
  contract_lock_in surfaces as the most *intense* (urgency 6.68).
  "overall_dissatisfaction" is held out as a baseline, not ranked. Pricing-pain
  evidence is clean SaaS pricing after the filter (no video-game quotes).
  Competitor map is normalized (HubSpot 49, merged from 4 spelling variants);
  the "what works well" section surfaces top positive_aspects (Easy to use 11x,
  customization, clean interface).
- `scripts/local_pr_review.sh` -> plan shape, plan/code consistency,
  `git diff --check`.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| `scripts/demand_opportunity_report.py` | ~375 |
| `tests/test_demand_opportunity_report.py` | ~120 |
| Plan doc (this update) | ~100 |
| **Total** | **~595** |

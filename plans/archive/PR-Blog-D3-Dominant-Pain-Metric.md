# PR-Blog-D3-Dominant-Pain-Metric

Ownership lane: `content-ops/blog-d3-dominant-pain-metric`

## Why this slice exists

Defect **D3** (pipedrive cluster): the "Where Pipedrive Users Feel the Most
Pain" section's prose ranked pain by FREQUENCY ("Pricing emerges as the dominant
pain category") while the pain-radar chart it introduces plots URGENCY -- where
UX (10.0) is the peak, not pricing (7.2). The prose also mislabeled the chart as
"by mention frequency", and a later line ("UX pain points appear less acute than
pricing or support friction") directly contradicted UX being the urgency peak.

Root cause: the `pain_analysis` section had NO `data_summary`, so the LLM
free-wrote the ranking; and `pain_data` took `signals[:6]` in frequency order
while the chart's `dataKey` is urgency -- so prose and chart used different
metrics.

## Scope (this PR)

- **Generator** (`_blueprint_vendor_deep_dive`, both byte-identical copies):
  sort signals by urgency before building `pain_data`, and add a deterministic
  `data_summary` that states the metric ("by reviewer urgency, 0-10") and the
  urgency-ranked top categories.
- **Data** (`pipedrive-deep-dive-2026-04.ts`): three sentences re-framed to the
  chart's urgency truth (UX 10.0 > pricing 7.2 > contract lock-in 6.8).

### Files touched

- `plans/PR-Blog-D3-Dominant-Pain-Metric.md`
- `atlas_brain/autonomous/tasks/b2b_blog_post_generation.py`
- `extracted_content_pipeline/autonomous/tasks/b2b_blog_post_generation.py`
- `tests/test_b2b_blog_post_generation.py`
- `atlas-churn-ui/src/content/blog/pipedrive-deep-dive-2026-04.ts`

## Mechanism

`sorted_signals = sorted(signals, key=avg_urgency, reverse=True)` feeds
`pain_data` (so the radar's first/peak entry is the highest-urgency category),
and the `pain_analysis` section gains a `data_summary`:
"Pain categories ranked by reviewer urgency (0-10). {top} shows the most acute
friction (urgency {n}), followed by: ...". The empty case is guarded by the
existing `if signals:`.

Data (urgency values read from the published chart, not assumed):
- L175: "by mention frequency ... most often" -> "by reviewer urgency ... most
  acute friction concentrates."
- L177: "Pricing emerges as the dominant ..." -> "UX friction emerges as the
  most acute pain category (urgency 10.0), followed by pricing (7.2) and
  contract lock-in (6.8). Data migration, API limitations, and support concerns
  appear at lower intensity." (same shape, urgency-correct ranking)
- L184: "UX pain points appear less acute than pricing or support friction ..."
  -> "UX pain points carry the highest urgency in the radar ..." (the Software
  Advice quote + "feature gaps" conclusion are unchanged).

## Intentional

- **Urgency is the canonical metric for this section.** The chart is hardwired
  to urgency; "most pain" maps to intensity, not count; and the frequency view
  is a different chart. So prose + chart + data_summary all rank by urgency.
- **Read the chart, not the recalled values.** Support is urgency 3.0 (lowest),
  not high -- it's frequently complained about but low-intensity; L183 (which is
  about complaint frequency/kind) correctly stays. L185 ("no single category
  dominates") stays -- 10/7.2/6.8/4.1/3.5/3.0 is a spread, not catastrophic.

## Deferred

- **D3-followup:** harmonize the OTHER pain view -- the frequency-ranked
  pain-category chart and its prose -- if both the urgency and frequency views
  remain in the post (design work, not this slice).
- **D4** (strengths/weaknesses chart mislabel).

## Verification

- New `test_deep_dive_pain_section_ranks_by_urgency_not_frequency`: signals where
  pricing is most frequent but ux most urgent -> the radar's first entry and the
  data_summary "most acute" are both ux, and the metric is labeled "reviewer
  urgency". Verified it FAILS on revert to the unsorted `signals[:6]` (radar[0]
  == "pricing").
- `pytest` generation + quote-gate suites -> 198 passed; both copies
  byte-identical.
- Data: no stale "dominant pain"/"mention frequency"/"less acute"; the post now
  reads UX-most-acute consistent with the chart; audit clean.

## Estimated diff size

| Area | LOC |
|---|---:|
| 2 generator copies (urgency sort + data_summary) | ~40 |
| Test | ~40 |
| pipedrive data (3 sentences) | ~6 |
| Plan doc | ~95 |
| **Total** | **~180** |

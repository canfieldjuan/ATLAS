# PR-Blog-D7b-Corpus-Data-Fix

Ownership lane: `content-ops/blog-d7b-corpus-data-fix`

## Why this slice exists

The generator fixes for **D7** shipped (PR-D7a #760 deep-dive total, PR-D7c #765
landscape category-scoping), but the already-PUBLISHED posts still carry the
overstated numbers. This is the data half (like #745 -> #752): correct the live
content to the scoped truth. Also removes the one real orphan PR #757's hardened
detector surfaced.

## Scope (this PR)

Content-field-only number corrections, recomputed from the live DB **within each
post's own stated analysis window** (not all-time -- the post claims "between
<dates>", so the honest fix keeps that window and corrects the counts inside it).
Verified each set is internally consistent (`verified + community == enriched`).

- **crm-landscape** (window 2026-02-25..2026-04-07, category-scoped): the
  mention-scoped corpus was inflated by off-category community cross-mentions.
  enriched 3,287 -> **1,054**; total collected 4,990 -> **2,022**; verified 172
  -> **147**; community 3,115 -> **907** (147+907=1,054); "total churn signals
  analyzed" 1,163 -> **1,054** (same windowed analyzed-corpus count, traced to
  `_fetch_category_topic_stats` enriched_reviews; the "2.1" avg urgency is
  windowed-correct, unchanged).
- **zoho-crm-deep-dive** (window 2026-02-28..2026-04-04, vendor-mention): the
  "collected" total used all sources. collected 940 -> **429**; enriched 268 ->
  **261**; verified 28 -> **24**; community 240 -> **237** (24+237=261); churn
  rate 5.2% -> **5.4%** (14/261; the 14-count is unchanged).
- **project-management-landscape**: delete the orphaned paragraph "The quote
  reflects Notion's appeal ..." -- it sits under a Wrike blockquote and there is
  NO Notion blockquote anywhere in the post (verified), so it dangles.

### Files touched

- `plans/PR-Blog-D7b-Corpus-Data-Fix.md`
- `atlas-churn-ui/src/content/blog/crm-landscape-2026-04.ts`
- `atlas-churn-ui/src/content/blog/zoho-crm-deep-dive-2026-04.ts`
- `atlas-churn-ui/src/content/blog/project-management-landscape-2026-04.ts`

## Mechanism

A one-off contextual transform (not committed) replaced full number-bearing
phrases (e.g. `(28 reviews)` -> `(24 reviews)`), never bare numbers, so chart
data is untouched; each substitution asserted its expected occurrence count.
Numbers trace to single windowed CTE queries per post against
`localhost:5433/atlas` (one row, all interdependent counts, same
allowlist+dedup+window). The orphan paragraph removal took the `<p>...</p>\n`
line out atomically.

## Intentional

- **Window-faithful, not all-time.** Each post claims a window; the corrected
  numbers are computed inside that same window, so "between <dates> we analyzed
  N" stays true rather than silently expanding the corpus claim to today.
- **Churn-signals line reconciled (review follow-up).** `1,163 total churn
  signals` was the all-time value of the same analyzed-corpus count
  (`_fetch_category_topic_stats` enriched_reviews); its windowed value is 1,054,
  identical to the corrected enriched count -- so it now reads consistently
  (1,054 enriched = 1,054 churn signals analyzed) instead of 1,163 > 1,054. The
  `2.1` avg urgency is windowed-correct (windowed == all-time == 2.1), unchanged.

## Deferred

- **D8** (`crm-landscape` "8 vendors" vs 7 charted / 5 profiled) -- the vendor
  count is a separate defect; left intact here.
- **D2/D3/D4** (pipedrive cluster).

## Verification

- Re-grep: zero stale numbers (3,287 / 4,990 / 3,115 / 940 / 268 / "(28
  reviews)" / "(240 reviews)" / 5.2%) remain; new numbers present and the
  enriched = verified + community identity holds in both posts.
- `audit-published-posts.js` across the corpus: `Orphaned quote reference 0/0`
  (was 1 -- the removed L266), `Form-prompt-as-quote 0/0`, empty-blockquote and
  markdown-in-`<p>` both 0. No new defects.
- Edits are content-field strings + one `<p>` removal; no field/schema/chart
  changes. (`npm run build` not run -- no node_modules in this checkout; the
  HTML-aware audit covers well-formedness.)

## Estimated diff size

| Area | LOC |
|---|---:|
| crm-landscape number corrections | ~12 |
| zoho-crm-deep-dive number corrections | ~14 |
| project-management-landscape (orphan removal) | ~1 |
| Plan doc | ~90 |
| **Total** | **~115** |

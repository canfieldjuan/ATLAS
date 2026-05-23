# PR-Blog-SEO-Keyword-Sample-Wins

Ownership lane: `content-ops/blog-seo-keyword-sample-wins`

## Why this slice exists

The SEO-keyword roadmap phase mines real buyer vocabulary from `b2b_reviews` to
augment the 78 posts' existing (generic, head-term) keyword sets. Two 5-post samples
validated the method (mine allowlist `review_text` -> Google-autocomplete demand-check
-> intent-classify). This slice commits ONLY the autocomplete-validated, right-intent
wins from those samples — concrete value locked in before the full sweep — without
touching anything unvalidated.

## Scope (this PR)

Append validated keyword candidates to `secondary_keywords` on the 6 sample posts that
HAD a new validated win. 9 additions total. No `target_keyword` changes; existing
secondary keywords retained (purely additive).

- salesforce-deep-dive: `salesforce too expensive`, `salesforce learning curve`
- crm-landscape: `CRM implementation cost`, `CRM setup cost`
- why-teams-leave-slack: `Slack notification overload`, `too many Slack notifications`
- real-cost-of-shopify: `shopify shipping costs too high`
- best-hr-hcm-for-51-200: `employee self service software`
- tableau-deep-dive: `tableau pricing`

Each was confirmed as a real search via Google autocomplete AND classified as
buyer/churn/comparison/cost intent (not career/how-to/navigational/status/troubleshooting).

### Files touched

- `plans/PR-Blog-SEO-Keyword-Sample-Wins.md`
- `atlas-churn-ui/src/content/blog/salesforce-deep-dive-2026-04.ts`
- `atlas-churn-ui/src/content/blog/crm-landscape-2026-04.ts`
- `atlas-churn-ui/src/content/blog/why-teams-leave-slack-2026-04.ts`
- `atlas-churn-ui/src/content/blog/real-cost-of-shopify-2026-04.ts`
- `atlas-churn-ui/src/content/blog/best-hr-hcm-for-51-200-2026-04.ts`
- `atlas-churn-ui/src/content/blog/tableau-deep-dive-2026-04.ts`

## Mechanism

Each edit appends entries to the existing `secondary_keywords` string array on one
line. Casing matches each post's own convention (lowercase `salesforce`/`shopify`/
`tableau`; capitalized `CRM`/`Slack`). No prose, schema, or `target_keyword` changes.

## Intentional

- **Only the 6 posts with a validated win are touched.** The other 4 technical-sample
  posts are deliberately NOT edited: fortinet ("vs palo alto") and sentinelone ("vs
  CrowdStrike") already carry their validated best keyword, power-bi's "too expensive"
  returned NO autocomplete (it is the value-priced option), and aws's top mined signal
  ("outage") is a status/news-intent trap. Inventing edits there would violate the
  validated-only rule and is itself the technical-sample finding (technical posts are
  comparison-led and already well-keyworded).
- **Additive, not replacing.** Existing secondary keywords are kept; these are the
  primary risk-free improvement (a new long-tail secondary cannot hurt ranking).
- **`target_keyword` untouched.** Changing the primary target is a bigger SEO call
  reserved for the volume-validation step.

## Deferred

- **Volume magnitude.** Autocomplete proves these terms ARE searched; it does not rank
  their volume. Prioritization (which to promote to `target_keyword`) waits on Search
  Console / a keyword tool — a separate roadmap step.
- **Full 78-post sweep.** Split by buyer type (business-buyer full pipeline ~60 posts;
  technical-buyer light-touch ~15-18). Method + reusable miner + both sample reports
  live in the seo-geo-aeo-blog-post skill scripts dir (untracked, not in this PR).

Parked hardening: none new.

## Verification

- All 6 edits applied via an assert-exact-match script (1 match per file required);
  `git diff` shows 6 files, 9 keyword additions, single-line array changes only.
- No `target_keyword` lines in the diff; no prose/content lines in the diff.
- Each committed keyword has a recorded autocomplete result in the two SEO-keyword
  sample reports (kept in the seo-geo-aeo-blog-post skill scripts dir, outside this repo).

## Estimated diff size

| Area | LOC |
|---|---:|
| 6 post files (1 line each, +/-) | ~12 |
| Plan doc | ~75 |
| **Total** | **~87** |

# PR-Blog-SEO-Keywords-CRM

Ownership lane: `content-ops/blog-seo-keywords-crm`

## Why this slice exists

Second full-category batch of the business-buyer SEO-keyword sweep (after Marketing
Automation #874; method proven on the mini-samples + #868). Covers the remaining CRM
posts (salesforce-deep-dive + crm-landscape already shipped in #868). Same validated
pipeline: mine allowlist `review_text` -> frustration-framed Google autocomplete ->
intent-classify -> keep buyer/churn/comparison/cost winners.

## Scope (this PR)

11 validated keyword additions appended to `secondary_keywords` across 8 CRM posts.
Additive only; no `target_keyword` or prose changes.

- hubspot-deep-dive-2026-03: `hubspot too expensive`, `hubspot vs salesforce`
- hubspot-deep-dive-2026-04: `hubspot too expensive`
- zoho-crm-deep-dive: `zoho crm vs salesforce`
- pipedrive-deep-dive: `Pipedrive vs HubSpot`, `Pipedrive pricing`
- close-vs-zoho-crm: `close crm pricing`
- copper-deep-dive: `copper crm pricing`
- insightly-deep-dive: `Insightly vs Zoho CRM`
- best-crm-for-51-200: `CRM implementation cost`, `CRM setup cost`

### Files touched

- `plans/PR-Blog-SEO-Keywords-CRM.md`
- `atlas-churn-ui/src/content/blog/hubspot-deep-dive-2026-03.ts`
- `atlas-churn-ui/src/content/blog/hubspot-deep-dive-2026-04.ts`
- `atlas-churn-ui/src/content/blog/zoho-crm-deep-dive-2026-04.ts`
- `atlas-churn-ui/src/content/blog/pipedrive-deep-dive-2026-04.ts`
- `atlas-churn-ui/src/content/blog/close-vs-zoho-crm-2026-04.ts`
- `atlas-churn-ui/src/content/blog/copper-deep-dive-2026-04.ts`
- `atlas-churn-ui/src/content/blog/insightly-deep-dive-2026-04.ts`
- `atlas-churn-ui/src/content/blog/best-crm-for-51-200-2026-04.ts`

## Mechanism

Each edit appends entries to the existing `secondary_keywords` array on one line,
applied via an assert-exact-match script (1 match per file). Casing matches each post.

## Intentional

- **Copper rescued via direct autocomplete, not mining.** The Copper review-text scope
  is contaminated by the metal (commodity/mining text), so mined phrases are unusable;
  `copper crm pricing` was validated by probing autocomplete directly. Same handling
  for the sparse Close scope.
- **Rejected** `hubspot learning curve` (career/course-collision; HubSpot is the easy
  tool) and `best crm for small business` for the 51-200 post (segment mismatch).
- **Additive, `target_keyword` untouched** — same discipline as #868/#874.

## Deferred

- **Volume magnitude** — autocomplete proves searched, not rank; primary-target
  promotion waits on Search Console / a keyword tool.
- **Remaining business-buyer categories** (~5: Project Management, Communication,
  E-commerce, Helpdesk, HR/HCM) + technical light-touch. Per-category records in the
  seo-geo-aeo-blog-post skill scripts dir.
- **switch-to-salesforce / switch-to-zoho-crm / top-complaint-every-crm /
  hubspot-vs-power-bi / insightly-vs-zoho-crm** — no new validated win (existing
  comparison/migration keyword already present or is the target_keyword).

Parked hardening: none new.

## Verification

- All 8 edits applied via assert-exact-match (1 match/file); `git diff` shows 8 posts,
  11 additions, single-line array changes only; no `target_keyword`/prose lines.
- Each committed keyword has a recorded autocomplete result (per-category record in the
  skill scripts dir).

## Estimated diff size

| Area | LOC |
|---|---:|
| 8 post files (1 line each, +/-) | ~16 |
| Plan doc | ~78 |
| **Total** | **~94** |

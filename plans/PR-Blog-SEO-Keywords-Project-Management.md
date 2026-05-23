# PR-Blog-SEO-Keywords-Project-Management

Ownership lane: `content-ops/blog-seo-keywords-project-management`

## Why this slice exists

Third full-category batch of the business-buyer SEO-keyword sweep (after Marketing
Automation #874 and CRM #875). Project Management is the richest review corpus (8,510
rows). Same validated pipeline: mine allowlist `review_text` -> frustration-framed
Google autocomplete -> intent-classify -> keep buyer/churn/comparison/cost winners.

## Scope (this PR)

11 validated keyword additions appended to `secondary_keywords` across 8 PM posts.
Additive only; no `target_keyword` or prose changes.

- asana-deep-dive: `Asana too expensive`, `Asana vs Monday`
- clickup-deep-dive: `ClickUp is overwhelming`, `ClickUp vs Asana`, `ClickUp learning curve`
- wrike-deep-dive: `Wrike pricing`
- jira-vs-trello: `jira too complex`
- jira-vs-mondaycom: `jira too complex`
- project-management-landscape: `project management software cost`
- switch-to-clickup-2026-03: `asana to clickup`
- switch-to-clickup-2026-04: `Asana to ClickUp`

### Files touched

- `plans/PR-Blog-SEO-Keywords-Project-Management.md`
- `atlas-churn-ui/src/content/blog/asana-deep-dive-2026-04.ts`
- `atlas-churn-ui/src/content/blog/clickup-deep-dive-2026-04.ts`
- `atlas-churn-ui/src/content/blog/wrike-deep-dive-2026-04.ts`
- `atlas-churn-ui/src/content/blog/jira-vs-trello-2026-03.ts`
- `atlas-churn-ui/src/content/blog/jira-vs-mondaycom-2026-04.ts`
- `atlas-churn-ui/src/content/blog/project-management-landscape-2026-04.ts`
- `atlas-churn-ui/src/content/blog/switch-to-clickup-2026-03.ts`
- `atlas-churn-ui/src/content/blog/switch-to-clickup-2026-04.ts`

## Mechanism

Each edit appends entries to the existing `secondary_keywords` array on one line, via
an assert-exact-match script (1 match per file). Casing matches each post.

## Intentional

- **Frustration-framed wins** drove the distinctive additions: `ClickUp is overwhelming`
  (#1 autocomplete; ClickUp's feature-bloat reputation) and `jira too complex` (Jira's
  complexity). `ClickUp learning curve` validated (df 44) where the easy tools don't.
- **Rejected** every `{vendor} api` (top raw signal almost everywhere — developer trap).
- **Additive, `target_keyword` untouched** — same discipline as #868/#874/#875.

## Deferred

- **Volume magnitude** — autocomplete proves searched, not rank; primary-target
  promotion waits on Search Console / a keyword tool.
- **Remaining business-buyer categories** (~4: Communication, E-commerce, Helpdesk,
  HR/HCM) + technical light-touch. Per-category records in the seo-geo-aeo-blog-post
  skill scripts dir.
- **teamwork-deep-dive** (sparse/conversational corpus, no clean win),
  **basecamp-deep-dive** (vs-asana already present), **best-project-management-for-201-1000
  / top-complaint-every-project-management / switch-to-asana** — no new validated win
  (would self-cannibalize the landscape cost term, or the keyword is already present).

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

# PR-Blog-SEO-Keywords-Communication

Ownership lane: `content-ops/blog-seo-keywords-communication`

## Why this slice exists

Fifth full-category batch of the business-buyer SEO-keyword sweep (after MA #874, CRM
#875, PM #877, E-commerce #879). Same validated pipeline: mine allowlist `review_text`
-> frustration-framed Google autocomplete -> intent-classify.

## Scope (this PR)

6 validated keyword additions appended to `secondary_keywords` across 3 Communication
posts. Additive only; no `target_keyword` or prose changes.

- slack-deep-dive: `Slack notification overload`, `too many Slack notifications`
- zoom-deep-dive: `zoom vs google meet`, `zoom too expensive`
- microsoft-teams-vs-notion: `Notion too expensive`, `Notion learning curve`

(The Slack terms were validated in the original sample for why-teams-leave-slack;
reused here for the general Slack deep-dive, which lacked them.)

### Files touched

- `plans/PR-Blog-SEO-Keywords-Communication.md`
- `atlas-churn-ui/src/content/blog/slack-deep-dive-2026-04.ts`
- `atlas-churn-ui/src/content/blog/zoom-deep-dive-2026-04.ts`
- `atlas-churn-ui/src/content/blog/microsoft-teams-vs-notion-2026-04.ts`

## Mechanism

Each edit appends entries to the existing `secondary_keywords` array on one line, via an
assert-exact-match script (1 match per file). Casing matches each post.

## Intentional

- **Lower-yield category, honestly scoped.** Communication is comparison-heavy with
  niche vs-posts; only 3 posts had a validated gap-filling win.
- **Rejected** `communication software cost` (autocomplete NONE — people search specific
  tools, not the category), `microsoft teams outage` (status-checking trap, like aws),
  and the `{vendor} api` signals (developer traps).
- **Additive, `target_keyword` untouched.**

## Deferred

- **Volume magnitude** — autocomplete proves searched, not rank.
- **Remaining business-buyer categories** (~2: Helpdesk, HR/HCM) + technical light-touch.
- **slack-vs-zoom, communication-landscape, top-complaint-every-communication,
  microsoft-teams-vs-salesforce** — no new validated win (comparison is the
  target_keyword, or the category/status terms don't validate).

Parked hardening: none new.

## Verification

- All 3 edits applied via assert-exact-match (1 match/file); `git diff` shows 3 posts,
  6 additions, single-line array changes only; no `target_keyword`/prose lines.
- Each committed keyword has a recorded autocomplete result (per-category record in the
  skill scripts dir).

## Estimated diff size

| Area | LOC |
|---|---:|
| 3 post files (1 line each, +/-) | ~6 |
| Plan doc | ~66 |
| **Total** | **~72** |

# PR-Blog-SEO-Keywords-Helpdesk

Ownership lane: `content-ops/blog-seo-keywords-helpdesk`

## Why this slice exists

Sixth full-category batch of the business-buyer SEO-keyword sweep (after MA #874, CRM
#875, PM #877, E-commerce #879, Communication #882). Same validated pipeline: mine
allowlist `review_text` -> frustration-framed Google autocomplete -> intent-classify.

## Scope (this PR)

4 validated keyword additions appended to `secondary_keywords` across 3 Helpdesk posts.
Additive only; no `target_keyword` or prose changes.

- intercom-deep-dive: `intercom too expensive`
- help-scout-vs-zendesk: `Zendesk too expensive`, `Zendesk alternatives`
- helpdesk-landscape: `help desk software cost`

### Files touched

- `plans/PR-Blog-SEO-Keywords-Helpdesk.md`
- `atlas-churn-ui/src/content/blog/intercom-deep-dive-2026-04.ts`
- `atlas-churn-ui/src/content/blog/help-scout-vs-zendesk-2026-04.ts`
- `atlas-churn-ui/src/content/blog/helpdesk-landscape-2026-04.ts`

## Mechanism

Each edit appends entries to the existing `secondary_keywords` array on one line, via an
assert-exact-match script (1 match per file). Casing matches each post.

## Intentional

- **Cost is the helpdesk buyer's nerve.** Intercom's pricing is infamous ("intercom
  pricing meme" autocompletes); Zendesk's cost drives the Help-Scout-as-cheaper-
  alternative angle (`Zendesk alternatives` validated rich). `intercom pricing` was NOT
  added — already present; only the distinct churn frame `intercom too expensive` is new.
- **Rejected** the `{vendor} api` developer signals.
- **Additive, `target_keyword` untouched.**

## Deferred

- **Volume magnitude** — autocomplete proves searched, not rank.
- **Remaining business-buyer category** (HR/HCM) + technical light-touch.
- **top-complaint-every-helpdesk** — no new validated win; existing vendor-complaint
  keywords appropriate.

Parked hardening: none new.

## Verification

- All 3 edits applied via assert-exact-match (1 match/file); `git diff` shows 3 posts,
  4 additions, single-line array changes only; no `target_keyword`/prose lines.
- Each committed keyword has a recorded autocomplete result (per-category record in the
  skill scripts dir).

## Estimated diff size

| Area | LOC |
|---|---:|
| 3 post files (1 line each, +/-) | ~6 |
| Plan doc | ~62 |
| **Total** | **~68** |

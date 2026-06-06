# PR-Blog-SEO-Keywords-HR-HCM

Ownership lane: `content-ops/blog-seo-keywords-hr-hcm`

## Why this slice exists

Seventh and final business-buyer category of the SEO-keyword sweep (after MA #874, CRM
#875, PM #877, E-commerce #879, Communication #882, Helpdesk #883). Same validated
pipeline: mine allowlist `review_text` -> frustration-framed Google autocomplete ->
intent-classify.

## Scope (this PR)

5 validated keyword additions appended to `secondary_keywords` across 2 HR/HCM posts.
Additive only; no `target_keyword` or prose changes.

- workday-deep-dive: `workday too expensive`, `workday implementation cost`
- hr-hcm-landscape: `BambooHR vs Gusto`, `Rippling vs Gusto`, `HR software cost`

### Files touched

- `plans/PR-Blog-SEO-Keywords-HR-HCM.md`
- `atlas-churn-ui/src/content/blog/workday-deep-dive-2026-04.ts`
- `atlas-churn-ui/src/content/blog/hr-hcm-landscape-2026-04.ts`

## Mechanism

Each edit appends entries to the existing `secondary_keywords` array on one line, via an
assert-exact-match script (1 match per file). Casing matches each post.

## Intentional

- **`workday implementation cost`** is the standout — Workday's implementation cost is a
  heavily-searched buyer concern ("average workday implementation cost", "how much does
  workday implementation cost"); the post had "workday implementation" but not the cost
  framing.
- **Pairwise comparisons** (`BambooHR vs Gusto`, `Rippling vs Gusto`) complement the
  landscape post's existing three-way "BambooHR vs Gusto vs Rippling" — distinct searches.
- **Rejected** `workday too complex` (autocomplete NONE — real but unsearched, like
  magento too complex).
- **Additive, `target_keyword` untouched.**

## Deferred

- **Volume magnitude** — autocomplete proves searched, not rank.
- **Technical light-touch pass** (Cybersecurity, Data & Analytics, Cloud Infra ~15-18
  posts) — comparison-led, lower yield; the only remaining sweep work.
- **gusto-vs-workday** — comparison is the target_keyword; no new validated win.
  best-hr-hcm-for-51-200 already shipped in #868 (`employee self service software`).

Parked hardening: none new.

## Verification

- All 2 edits applied via assert-exact-match (1 match/file); `git diff` shows 2 posts,
  5 additions, single-line array changes only; no `target_keyword`/prose lines.
- Each committed keyword has a recorded autocomplete result (per-category record in the
  skill scripts dir).

## Estimated diff size

| Area | LOC |
|---|---:|
| 2 post files (1 line each, +/-) | ~4 |
| Plan doc | ~62 |
| **Total** | **~66** |

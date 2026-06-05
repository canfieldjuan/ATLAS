# PR-Blog-SEO-Keywords-Technical

Ownership lane: `content-ops/blog-seo-keywords-technical`

## Why this slice exists

Final slice of the SEO-keyword sweep: the technical light-touch pass (Cybersecurity,
Data & Analytics, Cloud Infra), after the seven business-buyer categories (#868, #874,
#875, #877, #879, #882, #883, #885). Per the technical mini-sample, these posts are
comparison-led and mostly well-keyworded, so this is VERIFICATION + gap-filling: candidate
comparison/pricing terms probed directly against Google autocomplete (contamination-safe,
no review-text mining), adding only validated gaps.

## Scope (this PR)

8 validated keyword additions appended to `secondary_keywords` across 6 technical posts.
Additive only; no `target_keyword` or prose changes.

- looker-deep-dive: `Looker pricing`, `Looker vs Power BI`
- metabase-vs-tableau: `metabase vs power bi`, `metabase pricing`
- microsoft-defender-for-endpoint-deep-dive: `Microsoft Defender vs SentinelOne`
- amazon-web-services-deep-dive: `aws too expensive`
- azure-deep-dive: `Azure too expensive`
- linode-deep-dive: `Linode vs Vultr`

### Files touched

- `plans/PR-Blog-SEO-Keywords-Technical.md`
- `atlas-churn-ui/src/content/blog/looker-deep-dive-2026-04.ts`
- `atlas-churn-ui/src/content/blog/metabase-vs-tableau-2026-04.ts`
- `atlas-churn-ui/src/content/blog/microsoft-defender-for-endpoint-deep-dive-2026-04.ts`
- `atlas-churn-ui/src/content/blog/amazon-web-services-deep-dive-2026-04.ts`
- `atlas-churn-ui/src/content/blog/azure-deep-dive-2026-04.ts`
- `atlas-churn-ui/src/content/blog/linode-deep-dive-2026-04.ts`

## Mechanism

Each edit appends entries to the existing `secondary_keywords` array on one line, via an
assert-exact-match script (1 match per file). Casing matches each post.

## Intentional

- **Verification, not mining.** Technical review text is feature-noun-heavy and
  contamination-prone; candidate terms were validated by direct autocomplete instead.
- **Gaps were specific missing competitors** (Power BI as Looker's/Metabase's bigger
  rival, Vultr as Linode's third, SentinelOne as Defender's) **and the hyperscaler
  cost-shock frame** (aws/azure "too expensive" validated rich).
- **Already-covered posts left untouched**: fortinet, sentinelone, power-bi, tableau,
  switch-to-sentinelone, hubspot-vs-power-bi, azure-vs-salesforce, why-teams-leave-azure
  (×2) — their comparison + pricing terms are present.
- **Additive, `target_keyword` untouched.**

## Deferred

- **Volume magnitude** — autocomplete proves searched, not rank; `target_keyword`
  promotion (the next roadmap phase) waits on Search Console / a keyword tool.
- This completes the SEO-keyword sweep. Per-category records in the seo-geo-aeo-blog-post
  skill scripts dir.

Parked hardening: none new.

## Verification

- All 6 edits applied via assert-exact-match (1 match/file); `git diff` shows 6 posts,
  8 additions, single-line array changes only; no `target_keyword`/prose lines.
- Each committed keyword has a recorded autocomplete result (per-category record in the
  skill scripts dir).

## Estimated diff size

| Area | LOC |
|---|---:|
| 6 post files (1 line each, +/-) | ~12 |
| Plan doc | ~78 |
| **Total** | **~90** |

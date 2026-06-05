# PR-Blog-SEO-Keywords-Ecommerce

Ownership lane: `content-ops/blog-seo-keywords-ecommerce`

## Why this slice exists

Fourth full-category batch of the business-buyer SEO-keyword sweep (after Marketing
Automation #874, CRM #875, Project Management #877). Same validated pipeline: mine
allowlist `review_text` -> frustration-framed Google autocomplete -> intent-classify ->
keep buyer/churn/comparison/cost winners.

## Scope (this PR)

8 validated keyword additions appended to `secondary_keywords` across 7 E-commerce
posts. Additive only; no `target_keyword` or prose changes.

- shopify-deep-dive: `Shopify too expensive`, `Shopify transaction fees`
- real-cost-of-woocommerce: `WooCommerce extra fees`
- magento-deep-dive: `magento vs adobe commerce`
- switch-to-shopify-2026-03: `wix to shopify`
- switch-to-shopify-2026-04: `Wix to Shopify`
- switch-to-woocommerce: `shopify to woocommerce`
- top-complaint-every-e-commerce: `ecommerce platform cost`

### Files touched

- `plans/PR-Blog-SEO-Keywords-Ecommerce.md`
- `atlas-churn-ui/src/content/blog/shopify-deep-dive-2026-04.ts`
- `atlas-churn-ui/src/content/blog/real-cost-of-woocommerce-2026-04.ts`
- `atlas-churn-ui/src/content/blog/magento-deep-dive-2026-04.ts`
- `atlas-churn-ui/src/content/blog/switch-to-shopify-2026-03.ts`
- `atlas-churn-ui/src/content/blog/switch-to-shopify-2026-04.ts`
- `atlas-churn-ui/src/content/blog/switch-to-woocommerce-2026-04.ts`
- `atlas-churn-ui/src/content/blog/top-complaint-every-e-commerce-2026-04.ts`

## Mechanism

Each edit appends entries to the existing `secondary_keywords` array on one line, via an
assert-exact-match script (1 match per file). Casing matches each post.

## Intentional

- **Rejected real-but-unsearched and wrong-intent terms:** `woocommerce slow`
  (troubleshooting intent, not buyers), `magento performance issues` / `magento too
  complex` (both autocomplete NONE — genuine weaknesses, not search terms), `shopify app
  store` (navigational). The autocomplete gate earned its keep here.
- **Switch-direction migrations** (wix->shopify, shopify->woocommerce) fill real gaps.
- **Additive, `target_keyword` untouched** — same discipline as the prior sweep PRs.

## Deferred

- **Volume magnitude** — autocomplete proves searched, not rank; primary-target
  promotion waits on Search Console / a keyword tool.
- **Remaining business-buyer categories** (~3: Communication, Helpdesk, HR/HCM) +
  technical light-touch. Per-category records in the seo-geo-aeo-blog-post skill scripts dir.
- **woocommerce-deep-dive** — no new validated win (vs-shopify/pricing already present;
  "slow" is wrong intent). Not padded.

Parked hardening: none new.

## Verification

- All 7 edits applied via assert-exact-match (1 match/file); `git diff` shows 7 posts,
  8 additions, single-line array changes only; no `target_keyword`/prose lines.
- Each committed keyword has a recorded autocomplete result (per-category record in the
  skill scripts dir).

## Estimated diff size

| Area | LOC |
|---|---:|
| 7 post files (1 line each, +/-) | ~14 |
| Plan doc | ~76 |
| **Total** | **~90** |

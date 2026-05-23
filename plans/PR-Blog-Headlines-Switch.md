# PR-Blog-Headlines-Switch

Ownership lane: `content-ops/blog-headlines-switch`

## Why this slice exists

Headlines phase, scaling slice 4 / final (after sample #892, deep-dives #896, vs #897,
landscapes #898). Rewrites the high-N switch posts from the dry "Migration Guide: Why
Teams Are Switching to X" pattern to the approved punchier "Switching to X" style with
the review count surfaced. The two very low-N switch posts (switch-to-asana = 3 migrations,
switch-to-salesforce = 3 sources) are intentionally LEFT in their modest framing to avoid
overstating thin data.

## Scope (this PR)

7 `title` rewrites. No `seo_title`/body changes. Drops the "Migration Guide:" prefix,
leads with "Switching to X", surfaces each post's review count (from its description) and
names source platforms only where the description names them.

### Files touched

- `plans/PR-Blog-Headlines-Switch.md`
- `atlas-churn-ui/src/content/blog/switch-to-clickup-2026-03.ts`
- `atlas-churn-ui/src/content/blog/switch-to-klaviyo-2026-04.ts`
- `atlas-churn-ui/src/content/blog/switch-to-sentinelone-2026-04.ts`
- `atlas-churn-ui/src/content/blog/switch-to-shopify-2026-03.ts`
- `atlas-churn-ui/src/content/blog/switch-to-shopify-2026-04.ts`
- `atlas-churn-ui/src/content/blog/switch-to-woocommerce-2026-04.ts`
- `atlas-churn-ui/src/content/blog/switch-to-zoho-crm-2026-04.ts`

## Mechanism

Each edit replaces one `title:` line via an assert-exact-match script (1 match per file).
Counts are taken from each post's own description and verified: clickup-03 "645 enriched
reviews", klaviyo "638 Klaviyo reviews", sentinelone "488 SentinelOne reviews", shopify-03
"93 migration signals across 1,503 enriched reviews", shopify-04 "2383 Shopify reviews",
woocommerce "1467 reviews", zoho-crm "963 Zoho CRM reviews".

## Intentional

- **Sources named only when the description names them** — Klaviyo's "Mailchimp, Flodesk &
  MailerLite" are verbatim from its description. The others use a generic "why teams
  migrate" framing rather than asserting unverified source platforms.
- **Low-N posts left modest** — switch-to-asana / switch-to-salesforce keep "Migration
  Guide" framing (3 migrations / 3 sources); a punchy headline would overstate.
- **`seo_title` untouched.**

## Deferred

- **Headline phase COMPLETE** after this slice. Total rewritten across the phase: sample
  (5) + deep-dives (16) + vs (10) + landscapes (4) + switch (7) = 42 posts. Left as-is
  (already punchy): top-complaint (7), why-teams-leave (3), finding-led deep-dives (13),
  b2b-software/communication landscapes, urgency-gap vs-posts, low-N switch posts (2).
- Next roadmap phase: phased publish (then post-publish GSC volume refinement).

Parked hardening: none new.

## Verification

- All 7 edits applied via assert-exact-match (1 match/file); `git diff` = 7 `title` lines
  only; each surfaced count matches the post's description figure.

## Estimated diff size

| Area | LOC |
|---|---:|
| 7 post files (title) | ~14 |
| Plan doc | ~58 |
| **Total** | **~72** |

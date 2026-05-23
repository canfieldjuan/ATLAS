# PR-Blog-Headlines-Deep-Dives

Ownership lane: `content-ops/blog-headlines-deep-dives`

## Why this slice exists

Headlines phase, scaling slice 1 (after the user-approved sample #892). Rewrites the
16 deep-dive posts whose titles still use the dry, generic "Reviewer Sentiment Across N
Reviews" pattern, bringing them to the approved finding-led style. Posts that already
lead with a finding ("What N Reviews Reveal About X") are intentionally NOT touched.

## Scope (this PR)

16 `title` rewrites (one per post). No `seo_title`/body/number changes. Each new title
leads with the post's data-grounded angle (its validated keyword or a finding stated in
its own description), weaves the buyer concern, and preserves the exact review count.

### Files touched

- `plans/PR-Blog-Headlines-Deep-Dives.md`
- `atlas-churn-ui/src/content/blog/amazon-web-services-deep-dive-2026-04.ts`
- `atlas-churn-ui/src/content/blog/asana-deep-dive-2026-04.ts`
- `atlas-churn-ui/src/content/blog/basecamp-deep-dive-2026-04.ts`
- `atlas-churn-ui/src/content/blog/copper-deep-dive-2026-04.ts`
- `atlas-churn-ui/src/content/blog/fortinet-deep-dive-2026-04.ts`
- `atlas-churn-ui/src/content/blog/hubspot-deep-dive-2026-03.ts`
- `atlas-churn-ui/src/content/blog/hubspot-deep-dive-2026-04.ts`
- `atlas-churn-ui/src/content/blog/intercom-deep-dive-2026-04.ts`
- `atlas-churn-ui/src/content/blog/linode-deep-dive-2026-04.ts`
- `atlas-churn-ui/src/content/blog/looker-deep-dive-2026-04.ts`
- `atlas-churn-ui/src/content/blog/magento-deep-dive-2026-04.ts`
- `atlas-churn-ui/src/content/blog/sentinelone-deep-dive-2026-04.ts`
- `atlas-churn-ui/src/content/blog/tableau-deep-dive-2026-04.ts`
- `atlas-churn-ui/src/content/blog/workday-deep-dive-2026-04.ts`
- `atlas-churn-ui/src/content/blog/zoho-crm-deep-dive-2026-04.ts`
- `atlas-churn-ui/src/content/blog/zoom-deep-dive-2026-04.ts`

## Mechanism

Each edit replaces one `title:` line via an assert-exact-match script (1 match per
file). Numbers preserved exactly (comma formatting only) — verified: the multiset of
digits in removed vs added title lines is identical.

## Intentional

- **Angle is data-grounded, never invented.** Cost-led titles (AWS/Asana/HubSpot/
  Intercom "Too Expensive", Tableau/Looker/Copper/Workday pricing) use the validated
  keyword for that vendor. Finding-led titles use the post's own description: Linode
  ("After Akamai... pricing and uncertainty"), SentinelOne ("SMB pricing... EDR
  loyalty"). Basecamp (no validated cost angle) gets a safe praise/frustration framing.
- **`seo_title` left untouched** — already keyword-front ("{Vendor} Reviews 2026: ...").
- **Already-finding-led deep-dives not touched** (azure, brevo, clickup, mailchimp,
  pipedrive, power-bi, shopify, slack, teamwork, woocommerce, wrike, defender).

## Deferred

- Remaining headline slices: vs posts (dry "Comparing Reviewer Complaints Across N"),
  dry landscapes ("Compared by Real User Data"). Already-punchy posts (top-complaint,
  why-teams-leave, finding-led deep-dives) left as-is.

Parked hardening: none new.

## Verification

- All 16 edits applied via assert-exact-match (1 match/file); `git diff` = 16 `title`
  lines only; review-count multiset identical old vs new (no number altered).

## Estimated diff size

| Area | LOC |
|---|---:|
| 16 post files (title) | ~32 |
| Plan doc | ~58 |
| **Total** | **~90** |

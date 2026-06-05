# PR-Blog-Headlines-Vs

Ownership lane: `content-ops/blog-headlines-vs`

## Why this slice exists

Headlines phase, scaling slice 2 (after sample #892 and deep-dives #896). Rewrites the
10 head-to-head ("vs") posts whose titles use the dry "Comparing Reviewer Complaints
Across N Reviews" pattern (plus close-vs-zoho's flat "N Reviews Analyzed") to the
approved tension-led style. The two cross-category "Urgency Gap" vs-posts (azure-vs-
salesforce, notion-vs-salesforce) already carry a finding and are left as-is.

## Scope (this PR)

10 `title` rewrites (one per post). No `seo_title`/body/number changes. Each leads with
the comparison's deciding tension, grounded in that post's own description or validated
keyword; review counts preserved exactly.

### Files touched

- `plans/PR-Blog-Headlines-Vs.md`
- `atlas-churn-ui/src/content/blog/gusto-vs-workday-2026-04.ts`
- `atlas-churn-ui/src/content/blog/help-scout-vs-zendesk-2026-04.ts`
- `atlas-churn-ui/src/content/blog/hubspot-vs-power-bi-2026-04.ts`
- `atlas-churn-ui/src/content/blog/insightly-vs-zoho-crm-2026-04.ts`
- `atlas-churn-ui/src/content/blog/jira-vs-mondaycom-2026-04.ts`
- `atlas-churn-ui/src/content/blog/metabase-vs-tableau-2026-04.ts`
- `atlas-churn-ui/src/content/blog/microsoft-teams-vs-notion-2026-04.ts`
- `atlas-churn-ui/src/content/blog/microsoft-teams-vs-salesforce-2026-04.ts`
- `atlas-churn-ui/src/content/blog/slack-vs-zoom-2026-04.ts`
- `atlas-churn-ui/src/content/blog/close-vs-zoho-crm-2026-04.ts`

## Mechanism

Each edit replaces one `title:` line via an assert-exact-match script (1 match per file).
Numbers preserved exactly (comma formatting only) — verified: digit multiset identical
removed vs added.

## Intentional

- **Tension grounded in the post's data**, not invented: jira-vs-monday "Is Jira Too
  Complex?" (validated kw + desc "interface complexity"); ms-teams-vs-salesforce "Why
  Salesforce Frustrates More" (desc: Salesforce 3.2x higher urgency); metabase-vs-tableau
  "Open-Source Value vs Enterprise BI" (Metabase is open-source); ms-teams-vs-notion
  "Price & Learning Curve" (validated Notion kws + desc).
- **Low-N honesty kept** — help-scout-vs-zendesk (27) etc. keep the review count visible.
- **`seo_title` untouched**; the two "Urgency Gap" cross-category vs-posts left as-is.

## Deferred

- Last headline slice: dry landscapes ("Compared by Real User Data"). Already-punchy
  posts (top-complaint, why-teams-leave, finding-led deep-dives, switch posts) left as-is.

Parked hardening: none new.

## Verification

- All 10 edits applied via assert-exact-match (1 match/file); `git diff` = 10 `title`
  lines only; review-count multiset identical old vs new (no number altered).

## Estimated diff size

| Area | LOC |
|---|---:|
| 10 post files (title) | ~20 |
| Plan doc | ~52 |
| **Total** | **~72** |

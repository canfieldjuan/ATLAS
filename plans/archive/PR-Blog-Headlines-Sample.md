# PR-Blog-Headlines-Sample

Ownership lane: `content-ops/blog-headlines-sample`

## Why this slice exists

Headlines phase of the blog roadmap (after the SEO-keyword sweep + volume validation
deferred to GSC). Punchier headlines built from the validated keyword pool. This is the
SAMPLE slice — 5 representative posts (one per type), style approved by the user before
scaling to the rest — to de-risk a bulk style misfire.

## Scope (this PR)

Rewrite `title` (and `seo_title` where it sharpens the search result) on 5 posts. 7 field
edits. No body/prose/number changes beyond comma formatting; no `secondary_keywords` or
other field changes.

- salesforce-deep-dive: title + seo_title (lead with the price/learning-curve tension)
- crm-landscape: title + seo_title (lead with setup cost + churn)
- switch-to-clickup: title (name the source tools teams leave)
- real-cost-of-shopify: title (name the actual cost drivers)
- jira-vs-trello: title (lead with the "too complex" question)

### Files touched

- `plans/PR-Blog-Headlines-Sample.md`
- `atlas-churn-ui/src/content/blog/salesforce-deep-dive-2026-04.ts`
- `atlas-churn-ui/src/content/blog/crm-landscape-2026-04.ts`
- `atlas-churn-ui/src/content/blog/switch-to-clickup-2026-04.ts`
- `atlas-churn-ui/src/content/blog/real-cost-of-shopify-2026-04.ts`
- `atlas-churn-ui/src/content/blog/jira-vs-trello-2026-03.ts`

## Mechanism

Each edit replaces one `title:`/`seo_title:` line via an assert-exact-match script (1
match per edit). Style rules: lead with the sharpest finding/question, weave in the post's
validated keyword, preserve every number exactly (review counts, churn signals, complaint
counts), tension not clickbait.

## Intentional

- **Numbers preserved exactly** — `2256`->`2,256` / `1902`->`1,902` are comma formatting
  only, same values; counts traced to each post's own body. Given the correctness audit
  these posts went through, no number changes and no invented claims.
- **real-cost-of-shopify** title now reads "220 Complaints" (was "220 Reviews") — 220 is
  the complaint count per the post body (606 = reviews); the new wording is more accurate.
- **seo_title kept keyword-front** (Salesforce Reviews 2026 / CRM Software Comparison
  2026) so the search-result title still leads with the target term + year.

## Deferred

- **Scaling to the remaining ~73 posts** — pending the user's read on this sample landing.
- Body H1 sync — verified not needed (titles are not duplicated in post content; the H1
  renders from the `title` field).

Parked hardening: none new.

## Verification

- All 7 edits applied via assert-exact-match (1 match/edit); `git diff` shows 5 posts,
  7 title/seo_title line changes only; no body/number/keyword-field changes.

## Estimated diff size

| Area | LOC |
|---|---:|
| 5 post files (title/seo_title) | ~14 |
| Plan doc | ~70 |
| **Total** | **~84** |

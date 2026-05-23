# PR-Blog-SEO-Keywords-Marketing-Automation

Ownership lane: `content-ops/blog-seo-keywords-marketing-automation`

## Why this slice exists

First full-category batch of the business-buyer SEO-keyword sweep (the method was
proven on two mini-samples; sample-wins shipped in #868). This slice runs the
validated pipeline — mine allowlist `review_text` -> frustration-framed Google
autocomplete demand-check -> intent-classify -> keep only buyer/churn/comparison/cost
winners — across the Marketing Automation category and commits the validated wins.

## Scope (this PR)

8 validated keyword additions appended to `secondary_keywords` across 4 of the 5 MA
posts. Additive only; no `target_keyword` or prose changes.

- mailchimp-deep-dive: `Mailchimp too expensive`, `Mailchimp price increase`
- switch-to-klaviyo: `Klaviyo vs Omnisend`, `Klaviyo vs Mailchimp`, `Klaviyo too expensive`
- brevo-deep-dive: `Brevo vs Sendinblue`
- marketing-automation-landscape: `marketing automation cost`, `marketing automation pricing`

top-complaint-every-marketing-automation is intentionally NOT edited (no new validated
win; its existing vendor-complaint keywords are appropriate, and adding the landscape's
cost terms would self-cannibalize).

### Files touched

- `plans/PR-Blog-SEO-Keywords-Marketing-Automation.md`
- `atlas-churn-ui/src/content/blog/mailchimp-deep-dive-2026-04.ts`
- `atlas-churn-ui/src/content/blog/switch-to-klaviyo-2026-04.ts`
- `atlas-churn-ui/src/content/blog/brevo-deep-dive-2026-04.ts`
- `atlas-churn-ui/src/content/blog/marketing-automation-landscape-2026-04.ts`

## Mechanism

Each edit appends entries to the existing `secondary_keywords` string array on one
line, applied via an assert-exact-match script (1 match per file required). Casing
matches each post's own convention.

## Intentional

- **Only autocomplete-validated, right-intent terms committed.** Rejected the highest
  raw signal `mailchimp api` (df 53) as a developer/how-to trap, and `mailchimp
  learning curve` (autocomplete returned nothing — Mailchimp is the easy tool).
- **Additive, `target_keyword` untouched** — same discipline as #868; primary-target
  changes wait on volume validation.

## Deferred

- **Volume magnitude** — autocomplete proves searched, not rank; promotion to
  `target_keyword` waits on Search Console / a keyword tool.
- **Remaining business-buyer categories** (~6: CRM, Project Management, Communication,
  E-commerce, Helpdesk, HR/HCM) and the technical light-touch pass. Per-category
  records kept in the seo-geo-aeo-blog-post skill scripts dir (outside this repo).

Parked hardening: none new.

## Verification

- All 4 edits applied via assert-exact-match (1 match/file); `git diff` shows 4 posts,
  8 additions, single-line array changes only; no `target_keyword`/prose lines.
- Each committed keyword has a recorded autocomplete result (per-category record in the
  skill scripts dir).

## Estimated diff size

| Area | LOC |
|---|---:|
| 4 post files (1 line each, +/-) | ~8 |
| Plan doc | ~70 |
| **Total** | **~78** |

# PR-Blog-Fix-Orphaned-Quote-Refs

## Why this slice exists

Implements the data-fix phase of the orphaned-quote-reference cleanup planned
in #706. The #697 blockquote strip removed quotes but left 7 paragraphs across
5 posts that explicitly back-reference a now-deleted quote ("This quote
captures...", "quoted earlier", "The excerpt cuts off..."). These read as
non-sequiturs, are live in `main`, and are the last content blemish before
the affiliate posts are clean to publish. The SEO analyzer did not previously
detect this shape; a new `orphaned_quote_reference` detector (skill-side,
severity `high`) was added and is the verification surface for this fix.

## Scope (this PR)

Fix the 7 detector-confirmed orphaned references, preserving all analysis and
any inline quote -- drop only the dangling back-reference clause:

1. `clickup-deep-dive-2026-04` -- "...Group Director quoted earlier also
   noted..." -> "Reviewers also note..." (keeps the inline complaint quote).
2. `close-vs-zoho-crm-2026-04` (x2) -- "This quote reflects..." /
   "This quote illustrates..." -> lead with the analysis directly.
3. `marketing-automation-landscape-2026-04` -- "This quote captures the core
   tension:" -> "The core tension:".
4. `top-complaint-every-project-management-2026-04` -- drop "(quoted earlier)",
   keep the inline "the worst...so far" fragment + analysis.
5. `why-teams-leave-slack-2026-04` (x2) -- "The excerpt cuts off, but..." /
   "This excerpt shows..." -> lead with the signal.

Only the 7 ORPHANED references are touched. Sibling lines with the same wording
that DO have a preceding `<blockquote>` (close-vs-zoho 176/205, slack 67/110)
are legitimate and left untouched -- confirmed by the detector.

### Files touched

- `atlas-churn-ui/src/content/blog/clickup-deep-dive-2026-04.ts`
- `atlas-churn-ui/src/content/blog/close-vs-zoho-crm-2026-04.ts`
- `atlas-churn-ui/src/content/blog/marketing-automation-landscape-2026-04.ts`
- `atlas-churn-ui/src/content/blog/top-complaint-every-project-management-2026-04.ts`
- `atlas-churn-ui/src/content/blog/why-teams-leave-slack-2026-04.ts`
- `plans/PR-Blog-Fix-Orphaned-Quote-Refs.md`

## Mechanism

Content-field edits only. Each fix drops the explicit quote back-reference
clause and leads with the surrounding analysis (or keeps an inline quote where
present), so no real substance or evidence is lost and nothing is fabricated.
No `affiliate_url` / `seo` / `faq` / metadata fields are touched, so the
prerender, sitemap, and FTC-disclosure wiring are unaffected.

## Intentional

- **Only orphaned references.** The detector distinguishes a back-reference
  with no preceding blockquote (orphaned) from one with a real quote above it
  (legitimate). The latter are left as-is.
- **Drop the clause, keep the analysis.** Same judgment used for the lead-in
  trims (#697) and the copper repair (#692) -- preserve the point, remove the
  dangling pointer.
- **Generator-side prevention is a separate PR** (per #706), because it is a
  producer-behavior change with its own false-positive risk.

## Deferred

- Generator `_remove_unmatched_quote_lines` forward-sweep for quote
  back-reference follow-ons (the #706 Phase 3 PR).

## Verification

- `orphaned_quote_reference` detector across the corpus -> `0 / 0`
  (was `5 / 7`); the 4 "witness evidence" aggregate-reference posts that the
  rough scan mis-flagged are correctly NOT touched.
- Full audit -> `78 clean, 0 CRITICAL`.
- HTML well-formedness validator on all 5 edited posts -> 0 issues.
- `npm run build` -> `built in 4.74s`, `Pre-rendered 82 public routes`, no TS
  errors.
- `git diff --check` -> passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| 5 blog `.ts` posts (7 surgical clause edits) | ~12 |
| Plan doc | ~86 |
| **Total** | **~98** |

# PR-Blog-Fix-Form-Prompt-Quotes

## Why this slice exists

A per-post correctness audit (logged in `reports/blog-audit-findings.md`,
defect class **D1**) found G2-style review-FORM PROMPTS presented as genuine
reviewer quotes -- e.g. a blockquote reading *"What do you like best about
Pipedrive?"* attributed to a reviewer. These are the boilerplate questions the
G2 form asks; they get scraped into `review_text` and leak into blockquotes as
fake evidence (no opinion).

A new `form_prompt_quote` detector added to the `seo-geo-aeo-blog-post`
analyzer surfaced **47 occurrences across 31 posts** -- ~40% of the corpus.
This PR removes them.

## Scope (this PR)

For each form-prompt blockquote:
1. Remove the blockquote.
2. Remove its orphan lead-in (a preceding `<p>...:</p>` ending in a colon).
3. Remove a following paragraph ONLY if it explicitly references the prompt
   ("This question format ...", "This phrasing suggests a structured review
   prompt", "This <vendor> reviewer's question ..."). Standalone analysis
   follow-ons are KEPT (most of them) -- they do not depend on the quote.
4. Three follow-ons that referenced the removed quote with "This/The quote
   reflects ..." wording were rephrased to lead with the analysis (drop the
   dangling pointer), and one inline form-prompt in `power-bi-deep-dive`
   ("One Business Intelligence Analyst asked 'What do you like best about
   Microsoft Power BI'") was removed.

Item text and standalone analysis are preserved verbatim; nothing is
fabricated.

### Files touched

- `plans/PR-Blog-Fix-Form-Prompt-Quotes.md`
- `atlas-churn-ui/src/content/blog/basecamp-deep-dive-2026-04.ts`
- `atlas-churn-ui/src/content/blog/best-crm-for-51-200-2026-04.ts`
- `atlas-churn-ui/src/content/blog/best-hr-hcm-for-51-200-2026-04.ts`
- `atlas-churn-ui/src/content/blog/best-project-management-for-201-1000-2026-04.ts`
- `atlas-churn-ui/src/content/blog/close-vs-zoho-crm-2026-04.ts`
- `atlas-churn-ui/src/content/blog/crm-landscape-2026-04.ts`
- `atlas-churn-ui/src/content/blog/fortinet-deep-dive-2026-04.ts`
- `atlas-churn-ui/src/content/blog/gusto-vs-workday-2026-04.ts`
- `atlas-churn-ui/src/content/blog/hr-hcm-landscape-2026-04.ts`
- `atlas-churn-ui/src/content/blog/insightly-vs-zoho-crm-2026-04.ts`
- `atlas-churn-ui/src/content/blog/intercom-deep-dive-2026-04.ts`
- `atlas-churn-ui/src/content/blog/jira-vs-mondaycom-2026-04.ts`
- `atlas-churn-ui/src/content/blog/looker-deep-dive-2026-04.ts`
- `atlas-churn-ui/src/content/blog/marketing-automation-landscape-2026-04.ts`
- `atlas-churn-ui/src/content/blog/metabase-vs-tableau-2026-04.ts`
- `atlas-churn-ui/src/content/blog/microsoft-defender-for-endpoint-deep-dive-2026-04.ts`
- `atlas-churn-ui/src/content/blog/pipedrive-deep-dive-2026-04.ts`
- `atlas-churn-ui/src/content/blog/power-bi-deep-dive-2026-04.ts`
- `atlas-churn-ui/src/content/blog/project-management-landscape-2026-04.ts`
- `atlas-churn-ui/src/content/blog/salesforce-deep-dive-2026-04.ts`
- `atlas-churn-ui/src/content/blog/shopify-deep-dive-2026-04.ts`
- `atlas-churn-ui/src/content/blog/slack-deep-dive-2026-04.ts`
- `atlas-churn-ui/src/content/blog/switch-to-asana-2026-04.ts`
- `atlas-churn-ui/src/content/blog/switch-to-clickup-2026-04.ts`
- `atlas-churn-ui/src/content/blog/switch-to-salesforce-2026-04.ts`
- `atlas-churn-ui/src/content/blog/switch-to-woocommerce-2026-04.ts`
- `atlas-churn-ui/src/content/blog/top-complaint-every-e-commerce-2026-04.ts`
- `atlas-churn-ui/src/content/blog/top-complaint-every-helpdesk-2026-04.ts`
- `atlas-churn-ui/src/content/blog/top-complaint-every-project-management-2026-04.ts`
- `atlas-churn-ui/src/content/blog/woocommerce-deep-dive-2026-04.ts`
- `atlas-churn-ui/src/content/blog/zoom-deep-dive-2026-04.ts`

## Mechanism

A one-off transform (`/tmp/fix_form_prompts.py`, not committed) removes 45
form-prompt blockquotes + their orphan lead-ins, and 11 follow-ons that
explicitly reference the prompt. The follow-on removal is deliberately
conservative -- it only matches "this question / phrasing / snippet",
"reviewer's question", "structured prompt", "verified review structure/data",
"prompted questions" -- so the ~70% of follow-ons that are standalone analysis
survive. Content-field only; no `affiliate_url` / `seo` / `faq` / metadata
touched.

A re-scan then surfaced 4 residuals fixed by hand: 3 "This/The quote
reflects ..." follow-ons that dangled once their (form-prompt) blockquote was
removed (rephrased to lead with the analysis), and 1 inline form-prompt in
`power-bi-deep-dive` (sentence removed).

## Intentional

- **Remove the boilerplate, keep real analysis.** Form prompts carry no
  opinion; the surrounding standalone analysis does and is preserved.
- **Conservative follow-on removal.** Only prompt-referencing follow-ons are
  dropped; everything else stays, verified by re-scanning for orphaned
  references (0 after the residual fixes).
- **Content-field only**, uniform transform + 4 hand-fixed residuals.

## Deferred

- **Generator fix** -- filter form-prompt boilerplate from quote candidates so
  the shape cannot recur (the root cause).
- D2/D3/D4/D7/D8 -- catalogued, not yet addressed.

## Verification

- `seo-geo-aeo-blog-post` analyzer across the corpus -> `Form-prompt-as-quote
  0 / 0` (was `31 / 47`) and `Orphaned quote reference 0 / 0` (the 3 created by
  blockquote removal were fixed).
- HTML well-formedness validator on edited posts -> 0 issues.
- `npm run build` (atlas-churn-ui) -> `built in 4.64s`, `Pre-rendered 82
  public routes`, no TS errors.
- `git diff --check` -> passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| 31 blog `.ts` posts (45 form-prompt blockquotes + 11 follow-ons + 4 residuals removed) | ~470 |
| Plan doc | ~120 |
| **Total** | **~590** |

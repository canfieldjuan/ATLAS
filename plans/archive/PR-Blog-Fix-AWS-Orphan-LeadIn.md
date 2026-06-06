# PR-Blog-Fix-AWS-Orphan-LeadIn

Ownership lane: `content-ops/blog-fix-aws-orphan-leadin`

## Why this slice exists

Phase-2 fix (orphaned-lead-in triage). `amazon-web-services-deep-dive` L169
ended "...One reviewer describes a jarring incident:" with no blockquote
following; L170 then referenced "This $400 charge for minimal QuickSight usage"
-- i.e. the lead-in promised a quote that was stripped, and the next paragraph
dangled on it.

Recover-first (per the maintainer): the original review IS in the DB
(`b2b_reviews` id `559501ca`, "I got charged 400$ for Quicksight dashboard for no
reason, I have 3 charts there, I have raised support request and it is Unassigned
for 23 days") -- but its source is **Trustpilot**, which is EXCLUDED from the
blog source allowlist. Restoring it would quote an out-of-policy source. Per the
maintainer's call, **paraphrase** the real reported experience as a fact (no
source name), keeping the concrete detail without quoting the excluded platform.

## Scope (this PR)

- **Data** (`amazon-web-services-deep-dive-2026-04.ts` L169-170):
  - L169: drop the dangling "One reviewer describes a jarring incident:" clause.
  - L170: replace the "This $400 charge ..." (quote-dependent) prose with a
    grounded factual paraphrase: "One reviewer reported a $400 charge for a
    barely-used QuickSight dashboard, with a support request left unassigned for
    over three weeks -- illustrating how billing unpredictability compounds
    support frustration."

### Files touched

- `plans/PR-Blog-Fix-AWS-Orphan-LeadIn.md`
- `atlas-churn-ui/src/content/blog/amazon-web-services-deep-dive-2026-04.ts`

## Mechanism

The paraphrase is grounded verbatim-faithfully in review `559501ca` ($400 charge,
3 charts = "barely-used", support unassigned 23 days ~= "over three weeks"); no
invention, no excluded-source attribution. The dangling colon lead-in is removed.

## Intentional

- **Recover -> paraphrase (not restore).** The recovered quote is from an
  allowlist-excluded source (Trustpilot), so restoring it would re-introduce the
  source-policy inconsistency the corpus avoids. Paraphrasing keeps the real,
  DB-verified detail without quoting/naming the excluded platform.

## Deferred

- switch-to-asana orphaned lead-in (same pattern -- Software Advice, excluded;
  paraphrase slice).
- The rest of the ~20-post Phase-2 triage.

## Verification

- `audit-published-posts.js`: aws no longer flagged by `orphaned_quote_lead_in`
  (the new L170 is a statement, not a colon-terminated lead-in); 0 CRITICAL; no
  new defects.
- Paraphrase traces to `b2b_reviews` id `559501ca` (DB-verified, real).

## Estimated diff size

| Area | LOC |
|---|---:|
| aws data (lead-in drop + paraphrase) | ~4 |
| Plan doc | ~70 |
| **Total** | **~74** |

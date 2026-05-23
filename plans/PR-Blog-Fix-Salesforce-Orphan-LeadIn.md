# PR-Blog-Fix-Salesforce-Orphan-LeadIn

Ownership lane: `content-ops/blog-fix-salesforce-orphan-leadin`

## Why this slice exists

Phase-2 fix (first of the orphaned-lead-in triage from the new
`orphaned_quote_lead_in` detector). `salesforce-deep-dive` L159 ends with a
dangling quote lead-in -- "...One verified reviewer on G2 notes the platform's
ability to handle intricate workflows:" -- but no blockquote follows (L160 is
unrelated prose). The promised quote was stripped, leaving a colon-terminated
lead-in that reads as a non-sequitur.

This is the MECHANICAL one of the three flagged posts: L160 does not reference
the missing quote, so the lead-in clause is cleanly removable. (aws and
switch-to-asana have follow-ups that reference the stripped quote's specifics
and are handled as separate recover-first slices.)

## Scope (this PR)

- **Data** (`salesforce-deep-dive-2026-04.ts` L159): drop the trailing dangling
  sentence "One verified reviewer on G2 notes the platform's ability to handle
  intricate workflows:" -- the paragraph now ends at "...pre-built solutions for
  niche requirements." and flows into L160.

### Files touched

- `plans/PR-Blog-Fix-Salesforce-Orphan-LeadIn.md`
- `atlas-churn-ui/src/content/blog/salesforce-deep-dive-2026-04.ts`

## Mechanism

One sentence removed from the end of the L159 `<p>`. No quote to recover here
(none survives that the lead-in clearly maps to, and L160 doesn't depend on it),
so the honest fix is to drop the dangling promise rather than invent a quote.

## Intentional

- **Drop, not recover.** Unlike the other two orphaned lead-ins, the salesforce
  follow-up (L160) stands on its own, so no recovery/paraphrase is needed -- just
  remove the dangling clause.

## Deferred

- aws + switch-to-asana orphaned lead-ins -- separate recover-first slices (their
  follow-ups reference the stripped quote's $ figures).
- The rest of the ~20-post Phase-2 triage (undeclared-source, count-vs-list,
  prose-vs-chart).

## Verification

- `audit-published-posts.js`: `Orphaned quote lead-in` corpus count 3 -> 2
  (salesforce no longer flagged); 0 CRITICAL; no new defects.
- L159 now ends "...niche requirements." and flows into L160 ("That customization
  power comes with trade-offs...").

## Estimated diff size

| Area | LOC |
|---|---:|
| salesforce data (1 sentence drop) | ~2 |
| Plan doc | ~55 |
| **Total** | **~57** |

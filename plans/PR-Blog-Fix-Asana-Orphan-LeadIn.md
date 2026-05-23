# PR-Blog-Fix-Asana-Orphan-LeadIn

Ownership lane: `content-ops/blog-fix-asana-orphan-leadin`

## Why this slice exists

Phase-2 fix (last of the orphaned-lead-in triage). `switch-to-asana` L136 ended
"...One reviewer on Software Advice described an unexpected annual renewal
charge:" with no blockquote following; L137 dangled on it ("This quote doesn't
specify whether the $265 charge was for Asana or a previous tool...").

Recover-first (per the maintainer): the original review IS in the DB -- a
`software_advice` review ("...considering Asana ... opting into an annual plan. I
was just charged $265 for another year of a service I barely used") plus a
`trustpilot` variant. Both sources are EXCLUDED from the blog allowlist, so per
the maintainer's call, **paraphrase** the real reported experience as a fact (no
source name) rather than restore an out-of-policy quote.

Note: the real review clearly attributes the $265 to **Asana**, so the post's
hedge ("doesn't specify whether ... Asana or a previous tool") was inaccurate;
the paraphrase corrects it.

## Scope (this PR)

- **Data** (`switch-to-asana-2026-04.ts` L136-137):
  - L136: drop the dangling "One reviewer on Software Advice described an
    unexpected annual renewal charge:" clause.
  - L137: replace the quote-dependent prose with a grounded paraphrase: "One
    reviewer reported being charged $265 for an annual Asana renewal of a
    service they'd barely used, illustrating a recurring pattern of pricing
    backlash around auto-renewals and inflexible cancellation policies. This
    frustration tends to surface immediately after annual renewal charges,
    particularly when users discover unexpected billing after reduced usage."

### Files touched

- `plans/PR-Blog-Fix-Asana-Orphan-LeadIn.md`
- `atlas-churn-ui/src/content/blog/switch-to-asana-2026-04.ts`

## Mechanism

Paraphrase grounded in the real `software_advice` Asana-renewal review ($265,
annual, "barely used"); the inaccurate "doesn't specify Asana vs prior tool"
hedge is dropped (the review specifies Asana). No invention, no excluded-source
attribution, no "this quote" framing.

## Intentional

- **Recover -> paraphrase (not restore).** Both recovered sources (Software
  Advice, Trustpilot) are allowlist-excluded; paraphrasing keeps the real,
  DB-verified detail without quoting the out-of-policy source.
- **Corrected the hedge.** The grounded review attributes the charge to Asana,
  so the paraphrase states that rather than the original's incorrect "doesn't
  specify" caveat.

## Deferred

- The rest of the ~20-post Phase-2 triage (undeclared-source, count-vs-list,
  prose-vs-chart). This slice completes the orphaned-lead-in class (with
  salesforce #823 + aws #825).

## Verification

- `audit-published-posts.js`: switch-to-asana cleared on BOTH
  `orphaned_quote_lead_in` (lead-in gone) AND `undeclared_quoted_source`
  (corpus 16 -> 15 -- the dropped "on Software Advice" attribution was the only
  Software Advice mention); 0 CRITICAL; no new defects.
- Paraphrase traces to the real `software_advice` $265 Asana-renewal review.

## Estimated diff size

| Area | LOC |
|---|---:|
| switch-to-asana data (lead-in drop + paraphrase) | ~4 |
| Plan doc | ~75 |
| **Total** | **~79** |

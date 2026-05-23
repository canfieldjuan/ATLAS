# PR-Blog-DD1-Pain-As-Strength-B

Ownership lane: `content-ops/blog-dd1-pain-as-strength-b`

## Why this slice exists

Second (final) slice of the deep-pass DD1 fix (after #844, batch A). The
`pain_as_strength` detector found 19 deep-dives where the catch-all PAIN signal
`overall_dissatisfaction` sits in the strengths-weaknesses chart's STRENGTHS
series (spurious satisfaction score). #844 handled the 6 that also narrate it as
"overall satisfaction" in prose. This slice handles the remaining 13, which carry
the CHART defect but do NOT misnarrate it -- so they're chart-only moves.

## Scope (this PR)

Move `overall_dissatisfaction` from the strengths series to weaknesses in the
strengths-weaknesses chart of 13 posts (count in parens):
basecamp (88), brevo (72), fortinet (307), insightly (17), intercom (79),
magento (363), mailchimp (224), microsoft-defender-for-endpoint (2),
sentinelone (178), tableau (265), teamwork (47), wrike (80), zoho-crm (186).

No prose changes: a targeted scan confirmed none of the 13 narrate
`overall_dissatisfaction` as "overall satisfaction." (brevo's prose mentions
SUPPORT as a strength -- a legit-computed score, not the bug; zoho-crm L195
already calls overall_dissatisfaction "generalized frustration," consistent with
its now-corrected weakness placement.)

### Files touched

- `plans/PR-Blog-DD1-Pain-As-Strength-B.md`
- `atlas-churn-ui/src/content/blog/basecamp-deep-dive-2026-04.ts`
- `atlas-churn-ui/src/content/blog/brevo-deep-dive-2026-04.ts`
- `atlas-churn-ui/src/content/blog/fortinet-deep-dive-2026-04.ts`
- `atlas-churn-ui/src/content/blog/insightly-deep-dive-2026-04.ts`
- `atlas-churn-ui/src/content/blog/intercom-deep-dive-2026-04.ts`
- `atlas-churn-ui/src/content/blog/magento-deep-dive-2026-04.ts`
- `atlas-churn-ui/src/content/blog/mailchimp-deep-dive-2026-04.ts`
- `atlas-churn-ui/src/content/blog/microsoft-defender-for-endpoint-deep-dive-2026-04.ts`
- `atlas-churn-ui/src/content/blog/sentinelone-deep-dive-2026-04.ts`
- `atlas-churn-ui/src/content/blog/tableau-deep-dive-2026-04.ts`
- `atlas-churn-ui/src/content/blog/teamwork-deep-dive-2026-04.ts`
- `atlas-churn-ui/src/content/blog/wrike-deep-dive-2026-04.ts`
- `atlas-churn-ui/src/content/blog/zoho-crm-deep-dive-2026-04.ts`

## Mechanism

One chart datum per post: `{"name": "overall_dissatisfaction", "strengths": N,
"weaknesses": 0}` -> `{... "strengths": 0, "weaknesses": N}`. Same count, now in
the truthful series. Applied with a precise scripted swap (matched only the
overall_dissatisfaction entry with strengths>0/weaknesses==0), verified per the
detector and diff.

## Intentional

- **Only `overall_dissatisfaction` flipped.** pricing/support/onboarding in
  strengths are legitimately-computed scores (DB-verified band), left as-is.
- **No prose edits.** Verified the 13 don't misnarrate the moved category; forcing
  prose changes would be out of scope.

## Deferred

- DD2 (fabricated "Ira"/incoherent multipliers) and DD3 (widen prose_vs_chart for
  "X dominates") -- next, after DD1.

## Verification

- `pain_as_strength` detector: all 13 cleared (corpus run shows the only remaining
  flags are the 6 batch-A posts handled by the open #844 -- this branch is off
  origin/main). `--self-test` ALL PASS.
- `git diff` is exactly the chart entry per post (4 lines each, 26/26).
- Note: mailchimp/intercom/teamwork/tableau/zoho-crm file-overlap the open
  #837/#838/#840 on different regions (excluded-quote/Slashdot edits vs the chart
  datum) -- auto-merge.

## Estimated diff size

| Area | LOC |
|---|---:|
| 13 posts (chart move, 2 lines each) | ~52 |
| Plan doc | ~70 |
| **Total** | **~122** |

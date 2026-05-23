# PR-Blog-DD1-Prose-Leak

Ownership lane: `content-ops/blog-dd1-prose-leak`

## Why this slice exists

Recon of the un-swept post types found that the DD1 defect (the catch-all
`overall_dissatisfaction` count presented as positive "overall satisfaction")
leaks into the PROSE of non-deep-dive posts. The `pain_as_strength` detector is
CHART-based, so it only caught deep-dives; switching / vs / landscape / real-cost
/ best-X posts (many with no strengths-weaknesses chart) kept the false framing in
prose -- e.g. why-teams-leave-slack "293 mentions of positive overall satisfaction"
(293 is Slack's overall_dissatisfaction, confirmed in #844).

## Scope (this PR)

New PROSE detector `overall_satisfaction_prose` (untracked auditor): flags
"overall satisfaction" EXCEPT the legitimate rating-derived uses ("overall
satisfaction is high/solid (4.5 rating)", "... scores") and complaint framing.
Found 16 posts; all reframed. Fix pattern: drop the false "overall satisfaction"
and point retention/strength claims at the post's REAL strengths (UX, onboarding,
integration, features) -- never invent.

- **Count-cited leaks** (the count == the vendor's overall_dissatisfaction):
  why-teams-leave-slack (293), slack-vs-zoom (293), hubspot-vs-power-bi (534),
  switch-to-asana (208), switch-to-clickup (252), metabase-vs-tableau (265, x5).
- **Strength/retention-anchor framing** of the catch-all: crm-landscape,
  project-management-landscape, best-project-management, best-crm (x3),
  best-hr-hcm, hr-hcm-landscape, marketing-automation-landscape (L248 bullet),
  microsoft-teams-vs-notion, real-cost-of-woocommerce.
- **real-cost-of-copper**: cited "928 mentions of overall satisfaction" -- DB-
  verified Copper's overall_dissatisfaction is 194, so 928 is a misattributed
  count (no "overall_satisfaction" signal exists); dropped the false count + the
  derived "10:1 ratio" claim.
- **LEFT untouched (legitimate)**: helpdesk-landscape + marketing-automation L123
  "overall satisfaction is high / scores (4.x rating)" -- rating-derived, real.

### Files touched

- `plans/PR-Blog-DD1-Prose-Leak.md`
- `atlas-churn-ui/src/content/blog/best-crm-for-51-200-2026-04.ts`
- `atlas-churn-ui/src/content/blog/best-hr-hcm-for-51-200-2026-04.ts`
- `atlas-churn-ui/src/content/blog/best-project-management-for-201-1000-2026-04.ts`
- `atlas-churn-ui/src/content/blog/crm-landscape-2026-04.ts`
- `atlas-churn-ui/src/content/blog/hr-hcm-landscape-2026-04.ts`
- `atlas-churn-ui/src/content/blog/hubspot-vs-power-bi-2026-04.ts`
- `atlas-churn-ui/src/content/blog/marketing-automation-landscape-2026-04.ts`
- `atlas-churn-ui/src/content/blog/metabase-vs-tableau-2026-04.ts`
- `atlas-churn-ui/src/content/blog/microsoft-teams-vs-notion-2026-04.ts`
- `atlas-churn-ui/src/content/blog/project-management-landscape-2026-04.ts`
- `atlas-churn-ui/src/content/blog/real-cost-of-copper-2026-04.ts`
- `atlas-churn-ui/src/content/blog/real-cost-of-woocommerce-2026-04.ts`
- `atlas-churn-ui/src/content/blog/slack-vs-zoom-2026-04.ts`
- `atlas-churn-ui/src/content/blog/switch-to-asana-2026-04.ts`
- `atlas-churn-ui/src/content/blog/switch-to-clickup-2026-04.ts`
- `atlas-churn-ui/src/content/blog/why-teams-leave-slack-2026-04.ts`

## Mechanism

Each "overall satisfaction" claim is reframed to the post's real strengths or
removed (count-cited ones where the count is the dissatisfaction figure are
dropped, since there is no `overall_satisfaction` signal in the data). No
invention; counts verified against the DB (`overall_dissatisfaction` per vendor).

## Intentional

- **Prose-based detector** because the chart-based `pain_as_strength` structurally
  cannot see chartless posts -- this closes that blind spot as a durable gate.
- **Rating-derived "overall satisfaction is high (4.5 rating)" preserved** -- a
  real rating-based statement, not the dissatisfaction-count inversion.
- **Copper 928 dropped, not relabeled** -- it is not the overall_dissatisfaction
  count (194) and no overall_satisfaction signal exists, so the specific count is
  unfounded.

## Deferred

- Other pre-D7 derived numbers in landscape posts (per-vendor sample sizes,
  churn-signal counts, chart signal_counts) -- parked.

Parked hardening: none new.

## Verification

- `overall_satisfaction_prose` detector added (untracked auditor); `--self-test`
  ALL PASS (3 fixtures incl. a rating-based not-flagged guard).
- Full corpus (78 posts): ALL detectors = 0 -- `overall_satisfaction_prose` 0 and
  no regression in the other 8 classes.

## Estimated diff size

| Area | LOC |
|---|---:|
| 16 posts (prose reframes) | ~66 |
| Plan doc | ~95 |
| **Total** | **~161** |

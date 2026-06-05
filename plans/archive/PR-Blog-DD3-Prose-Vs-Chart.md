# PR-Blog-DD3-Prose-Vs-Chart

Ownership lane: `content-ops/blog-dd3-prose-vs-chart`

## Why this slice exists

DD3, the last detector-flagged deep-pass class. The widened
`detectProseVsChartMetric` (catches "X dominates", not just "the dominant pain")
flags 3 posts where prose calls a category the dominant pain but the pain-radar
peaks elsewhere: asana, linode, mailchimp (microsoft-defender + workday were
already fixed in #831). DB-grounded per-post reframes -- NOT a uniform "swap in
the chart-top", because the chart top is sometimes a low-frequency outlier
(asana data_migration=6.8 from only 14 mentions -- the parked chart-provenance
item).

## Scope (this PR)

DB-verified per-category frequency drove each reframe:
- **asana** (chart-top data_migration is a 14-mention outlier; pricing is 4th by
  frequency, 48): L192 "Pricing dominates. Pricing complaints far exceed other
  categories" -> "Pricing is a leading churn driver. ..." (drop the false
  frequency/dominance superlatives; keep the post's renewal-shock churn thesis).
  Also L176 "The most acute pain signal centers on ..." -> "A central pain
  signal ..." (the variant superlative the narrow detector misses; data_migration
  is the urgency top, not pricing).
- **linode** L165: "Overall dissatisfaction dominates the pain landscape" ->
  "Overall dissatisfaction is the most frequently cited pain category" (grounded:
  it is the top WEAKNESS by frequency -- 42 -- since pricing is classified as a
  strength; chart-top is Ux by urgency).
- **mailchimp** L123/L158/L183: "Pricing dominates / is the dominant pain point"
  -> "the most frequently cited specific complaint" / "leads complaint volume
  among the specific pain categories" (grounded: pricing is #1 among SPECIFIC
  categories at 96; overall_dissatisfaction 139 is the catch-all; chart-top is
  Support by urgency).

### Files touched

- `plans/PR-Blog-DD3-Prose-Vs-Chart.md`
- `atlas-churn-ui/src/content/blog/asana-deep-dive-2026-04.ts`
- `atlas-churn-ui/src/content/blog/linode-deep-dive-2026-04.ts`
- `atlas-churn-ui/src/content/blog/mailchimp-deep-dive-2026-04.ts`

## Mechanism

Frequency/churn reframe (the #831 pattern): the prose stops claiming the named
category is THE top pain by the chart's (urgency) measure, and instead states what
the data supports -- most-frequent (linode/mailchimp) or the churn driver (asana).
This clears the detector AND avoids crowning a provenance-suspect urgency outlier
(asana data_migration). No invention; reframes are DB-grounded.

## Intentional

- **No reorder / no crowning the urgency-top for asana.** Its chart-top
  (data_migration 6.8) is a 14-mention outlier (parked chart-provenance), so
  elevating it would be less truthful than dropping the superlative.
- **"Specific" qualifier for mailchimp/linode** where the catch-all
  overall_dissatisfaction would otherwise be the literal volume leader.

## Deferred

- The pain-radar **chart-provenance** question (urgency values vs naive aggregate)
  remains parked in ATLAS-HARDENING -- this slice fixes prose<->chart agreement,
  not chart-truth.
- Recon on the un-swept post types (landscape / top-complaint / why-teams-leave).

Parked hardening: none new (chart-provenance already parked).

## Verification

- Widened `detectProseVsChartMetric` `--self-test`: ALL PASS.
- All 3 posts re-audited (`--slug=`): clean. Full corpus (78 posts):
  `prose_vs_chart_metric` = **0** -- the class is complete corpus-wide. With this +
  the excluded-source follow-up (#854), every detector class is now 0 corpus-wide.

## Estimated diff size

| Area | LOC |
|---|---:|
| 3 posts (asana 2, linode 1, mailchimp 3) | ~12 |
| Plan doc | ~70 |
| **Total** | **~82** |

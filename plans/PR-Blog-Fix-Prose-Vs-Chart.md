# PR-Blog-Fix-Prose-Vs-Chart

Ownership lane: `content-ops/blog-fix-prose-vs-chart`

## Why this slice exists

Phase-2 fix for the `prose_vs_chart_metric` class: prose names a category as the
"dominant / most acute" pain, but the pain-radar chart (which plots
`avg_urgency_when_mentioned`) peaks on a different category. Two posts flagged;
the chart is treated as the authoritative reader-facing artifact and the prose is
made consistent with it.

- **microsoft-defender L147** -- "Integration emerges as **the dominant pain
  category**" but the radar peaks at Security (3.5); Integration is 3rd (2.0).
  DB-verified, Integration is not dominant by urgency OR frequency, so no
  frequency-reframe is available -- just drop the false superlative. The post's
  FAQ (L112) already frames integration as "dominate[s] reviewer complaints"
  (qualitative), and L149/L150 make no ranking superlative, so this is a
  one-sentence fix.
- **workday L180-190** -- the detector flagged L183 ("Support erosion is **the
  dominant pain category**"), but the WHOLE pain-radar section narrated the
  urgency chart with the ranking inverted: Support called dominant (chart: 5th of
  6, 1.7), Overall Dissatisfaction called a "high score" / "second-largest"
  (chart: LOWEST, 1.1), technical_debt treated as a footnote (chart: HIGHEST,
  3.2). Per the maintainer's call, fix the whole section to match the chart.

## Scope (this PR)

- **microsoft-defender** L147: "emerges as the dominant pain category" ->
  "emerges as a recurring pain theme". No other change (rest of the section is
  qualitative and chart-consistent).
- **workday** L183-189: reorder the six already-present category paragraphs into
  the radar's urgency order (technical_debt -> data_migration -> integration ->
  security -> support -> overall_dissatisfaction) and reword the three inverted
  rank claims:
  - technical_debt -> "registers the highest complaint intensity on the radar".
  - support -> "registers lower on the intensity radar, but surfaces frequently"
    and is "a primary wedge driving negative sentiment" (frequency/churn-narrative
    note, NOT "dominant pain category"); matches FAQ L123.
  - overall_dissatisfaction -> "shows the lowest complaint intensity ... even as
    it appears the most often" (drops the chart-contradicting "high score" /
    "second-largest"). DB-grounded: overall_dissatisfaction is the most frequent
    pain (82 mentions) and lowest urgency (1.1).
  All real review content (support decline, mid-market nuance, technical-debt
  maintenance, data migration, integration points, security config) is preserved.
- **Park** (`ATLAS-HARDENING.md`): the pain-radar urgency values don't reproduce
  from a naive all-time aggregate (likely windowed/scorecard-derived); chart
  provenance is unverified -- a deep-pass item, pairs with the D3-followup
  frequency-view chart.

### Files touched

- `plans/PR-Blog-Fix-Prose-Vs-Chart.md`
- `atlas-churn-ui/src/content/blog/microsoft-defender-for-endpoint-deep-dive-2026-04.ts`
- `atlas-churn-ui/src/content/blog/workday-deep-dive-2026-04.ts`
- `ATLAS-HARDENING.md`

## Mechanism

Chart-as-authoritative: the reader sees the radar, so the prose is made to match
its urgency ranking. The workday support-erosion "wedge" claim survives because a
"wedge driving sentiment" is a churn-narrative lens distinct from pain urgency --
and the rewrite makes the body match the FAQ (L123) verbatim in framing. DB
queries (per-category frequency + urgency, generator filters) were used only to
RULE OUT a frequency-reframe (the named categories aren't frequency-dominant
either), not to assert new rankings.

## Intentional

- **No chart edits.** The chart values come from a windowed
  `avg_urgency_when_mentioned` computation not reproduced this slice; treat the
  chart as authoritative rather than declare it wrong (parked instead).
- **No generator edit.** Data-correction for already-published posts; the
  pain-radar prose is LLM-written narrative, not a deterministic generator field.
- **Workday "primary wedge" inherited, not DB-verified.** This slice is
  prose_vs_chart, not full grounding; the wedge thesis (matching the FAQ) is kept
  as the post's churn angle. If the deep pass disputes it, that's a grounding-pass
  call, not this slice.

## Deferred

- workday `undeclared_quoted_source` (L166 Slashdot quote) -- pre-existing, part
  of the separate 15-post undeclared-source class, untouched here.
- The pain-radar chart-provenance question (parked in ATLAS-HARDENING).
- Remaining Phase-2 triage: `undeclared_quoted_source` (15, needs scoping).

## Verification

- Auditor `--self-test`: ALL PASS (no detector change this slice).
- Full corpus (`--repo=atlas-churn-ui`, 78 posts): `prose_vs_chart_metric` = **0**
  (both posts cleared); microsoft-defender shows NO flags; workday shows only the
  pre-existing `undeclared_quoted_source=1` (L166, not introduced here -- `git
  diff` does not touch the Slashdot quote).
- Workday section reads in urgency order and each rank claim matches the radar
  (technical_debt highest 3.2 ... overall_dissatisfaction lowest 1.1).

## Estimated diff size

| Area | LOC |
|---|---:|
| microsoft-defender data (1 sentence) | ~2 |
| workday data (reorder + 3 rewordings) | ~12 |
| ATLAS-HARDENING park entry | ~8 |
| Plan doc | ~105 |
| **Total** | **~127** |

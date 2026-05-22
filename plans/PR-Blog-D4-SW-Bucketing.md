# PR-Blog-D4-SW-Bucketing

Ownership lane: `content-ops/blog-d4-sw-bucketing`

## Why this slice exists

Defect **D4** (last of the pipedrive cluster): the deep-dive "Strengths vs
Weaknesses" chart mislabeled complaint categories as strengths. The published
`pipedrive-deep-dive` showed `overall_dissatisfaction` (118), `pricing` (51),
`ux` (26), `features` (19), `data_migration` (7) as "strengths", and the prose
claimed "eight strength categories". Root cause: the signals-based fallback
(used when the product profile is thin) split pain-category signals into
strengths/weaknesses by an urgency threshold (`urgency < 3.0 -> strengths`) --
but pain signals are all complaints, so a low-urgency pain is not a strength.

This is data-untruthful output ("overall dissatisfaction" as a top strength), so
the chart-bucketing and the published numbers are fixed inline; the fallback's
inability to show TRUE strengths is design work, parked.

## Scope (this PR)

Thinnest real slice: make the strengths/weaknesses bucketing truthful.

- **Generator** (`_blueprint_vendor_deep_dive`, both byte-identical copies): in
  the signals fallback, bucket every pain category as a weakness (drop the
  urgency-based "strength" split).
- **Data** (`pipedrive-deep-dive-2026-04.ts`): flip the 5 pain entries
  (overall_dissatisfaction, pricing, ux, features, data_migration) from
  strengths to weaknesses; keep integration (18) + onboarding (10) as strengths
  (the prose + reviewer quotes back them); support stays a weakness. Fix the
  L163 prose count claim.

### Files touched

- `plans/PR-Blog-D4-SW-Bucketing.md`
- `ATLAS-HARDENING.md`
- `HARDENING.md`
- `atlas_brain/autonomous/tasks/b2b_blog_post_generation.py`
- `extracted_content_pipeline/autonomous/tasks/b2b_blog_post_generation.py`
- `tests/test_b2b_blog_post_generation.py`
- `atlas-churn-ui/src/content/blog/pipedrive-deep-dive-2026-04.ts`

## Mechanism

Fallback branch: `area_map[cat]["weaknesses"] += cnt` for every pain category
(no urgency split), sorted by weaknesses. The published chart was mixed-
provenance (pain categories wrongly as strengths, plus real strengths
integration/onboarding from another source), so the data fix is a hand-edit per
row, not a re-derivation: 5 pain rows flipped to weaknesses, integration +
onboarding kept as strengths, support unchanged -> 2 strengths + 6 weaknesses.
L163 "eight strength categories and two primary weakness areas" -> "two strength
categories (onboarding and integration) and six recurring weakness areas, led by
overall dissatisfaction and pricing complaints" -- truthful to the new chart and
consistent with the L165-166 strength prose (left unchanged, now agrees).

## Intentional

- **Pain signals can only yield weaknesses.** A low-urgency pain is still a
  pain; the urgency split was the bug.
- **Kept integration/onboarding as strengths.** The prose (L165-166) + the
  G2/Slashdot reviewer quotes independently establish them as strengths, so
  flipping them would create a NEW prose-vs-chart contradiction. Per-category,
  prose-anchored split, not a mechanical flip-all.

## Deferred

**Parked hardening** in `ATLAS-HARDENING.md` (this session uses a separate hardening file from
the root `HARDENING.md`, per the maintainer, to avoid colliding with the
concurrent content-ops-station sessions). Per the #797 Codex P2, root
`HARDENING.md` now carries a one-line pointer to `ATLAS-HARDENING.md` so the §3d
"scan HARDENING.md at slice start" step still surfaces these entries:

- The signals fallback cannot show TRUE strengths (one-sided chart) -- needs a
  separate strengths source. Effort M, polish.
- The "Strengths vs Weaknesses" title stays two-sided while fallback data is
  one-sided. Effort S, polish.

Also remaining in the broader audit: D2-followup (Zoho/Zoho CRM merge),
D3-followup (frequency-view pain chart), D5/D6 (minor single-instance).

## Verification

- New `test_deep_dive_sw_chart_buckets_pain_signals_as_weaknesses`: thin profile
  + pain signals (incl. a low-urgency `ux`) -> every chart entry has
  `strengths == 0` and `weaknesses > 0`. Verified it FAILS on revert to the
  urgency split (ux gets `strengths == 30`).
- `pytest` generation + quote-gate suites -> 198 passed; both copies
  byte-identical.
- Data: pipedrive chart now 2 strengths (integration, onboarding) + 6 weaknesses;
  no "eight strength categories" prose; audit clean.

## Estimated diff size

| Area | LOC |
|---|---:|
| 2 generator copies (fallback bucketing) | ~24 |
| Test | ~38 |
| pipedrive data (5 chart flips + L163) | ~12 |
| ATLAS-HARDENING.md (new file) + root HARDENING.md pointer | ~57 |
| Plan doc | ~95 |
| **Total** | **~230** |

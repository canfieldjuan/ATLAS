# PR-Blog-D8-Vendor-Count

Ownership lane: `content-ops/blog-d8-vendor-count`

## Why this slice exists

Defect **D8** (`reports/blog-audit-findings.md`): a landscape post states a vendor
count independently of the vendors it actually renders. `crm-landscape` claimed
"**8** vendors" (title, seo, faq, prose) while the churn-urgency chart shows
**7** (Copper, Freshsales, Zoho CRM, Close, Pipedrive, Insightly, Nutshell --
no Salesforce/HubSpot). Generator fix + the one published-post correction.

## Scope (this PR)

- **Generator** (`_blueprint_market_landscape`, both byte-identical
  `b2b_blog_post_generation.py` copies): the headline `vendor_count` came from
  `ctx["vendor_count"]` (an independent category-wide count); derive it instead
  from the vendors actually rendered in the urgency chart.
- **Data** (`crm-landscape-2026-04.ts`): correct the 7 published "8 vendors"
  claims to **7** (the charted count).

### Files touched

- `plans/PR-Blog-D8-Vendor-Count.md`
- `atlas_brain/autonomous/tasks/b2b_blog_post_generation.py`
- `extracted_content_pipeline/autonomous/tasks/b2b_blog_post_generation.py`
- `tests/test_b2b_blog_post_generation_quote_gate.py`
- `atlas-churn-ui/src/content/blog/crm-landscape-2026-04.ts`

## Mechanism

The urgency-chart loop (`urgency_data`, one entry per vendor with signals) is
moved above the section list so it is the single source of truth, and
`rendered_vendor_count = len(urgency_data) or vendor_count` replaces
`ctx["vendor_count"]` in the four headline surfaces (hook key_stats + summary,
takeaway stats, suggested title). The fallback to `vendor_count` keeps a
degenerate no-signals landscape from rendering "0 vendors". The pain-chart
`dataKey: "vendor_count"` (a chart column, different semantic) is untouched.

Data: contextual phrase replacements (never bare `8`) took the count from 8 -> 7
in all 7 spots (title, seo_title, description, seo_description, faq, two prose
lines). The seo vendor-name list ("Salesforce, HubSpot, ...") is left as-is --
that is editorial naming, a different defect class than the count.

## Intentional

- **Single source of truth.** Deriving from `len(urgency_data)` (not a separate
  `sum(... if signals)`) means the headline can't drift from the chart if the
  chart's inclusion rule later changes.
- **Count only, in the data fix.** Per the catalog, D8 is the count defect;
  the seo vendor-name list is out of scope.

## Deferred

- **D8b -- profile coverage gap:** the description's "vendor-by-vendor strengths
  and weaknesses" implies the full set, but only 5 of the 7 charted vendors get
  strength/weakness profiles. Complement of the count defect; separate slice.
- **D2/D3/D4** (pipedrive cluster).

## Verification

- New `test_market_landscape_headline_vendor_count_reflects_rendered_chart`:
  ctx says 8 but only 7 vendor_signals carry signals -> the hook reads 7
  (`key_stats["vendor_count"] == 7`, "7 major vendors" in summary). Verified it
  FAILS on revert (`rendered_vendor_count = vendor_count` -> `8 == 7`).
- `pytest` generation + quote-gate suites -> 194 passed; both copies
  byte-identical.
- Data: case-insensitive grep -> zero "8 vendor" claims remain; the 7 spots now
  read "7"; the `"vendor_count": 7` pain-chart values (already 7) untouched.

## Estimated diff size

| Area | LOC |
|---|---:|
| 2 generator copies (loop move + derived count) | ~55 |
| Test | ~35 |
| crm-landscape data (7 count corrections) | ~14 |
| Plan doc | ~90 |
| **Total** | **~195** |

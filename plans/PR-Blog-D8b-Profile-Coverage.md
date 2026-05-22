# PR-Blog-D8b-Profile-Coverage

Ownership lane: `content-ops/blog-d8b-profile-coverage`

## Why this slice exists

D8b (the coverage complement of D8 #775): `crm-landscape` charts 7 vendors but
renders strength/weakness profiles for only 5 (`vendor_profiles[:5]` cap), while
the description claimed "vendor-by-vendor strengths and weaknesses" and the prose
said "where each product holds strength or shows weakness" -- both implying every
vendor is profiled. Charted-but-unprofiled: Zoho CRM, Pipedrive.

## Scope (this PR)

- **Generator** (`_blueprint_market_landscape`, both byte-identical copies):
  surface `profile_count = min(len(vendor_profiles), 5)` in the hook key_stats,
  so the description can be framed honestly around the count actually rendered.
- **Data** (`crm-landscape-2026-04.ts`): soften the two over-claims to "the
  leading vendors". Leave the body's "vendor-by-vendor churn risk scores" -- that
  is the urgency chart (7 vendors), which is accurate.

### Files touched

- `plans/PR-Blog-D8b-Profile-Coverage.md`
- `atlas_brain/autonomous/tasks/b2b_blog_post_generation.py`
- `extracted_content_pipeline/autonomous/tasks/b2b_blog_post_generation.py`
- `tests/test_b2b_blog_post_generation_quote_gate.py`
- `atlas-churn-ui/src/content/blog/crm-landscape-2026-04.ts`

## Mechanism

`profile_count = min(len(vendor_profiles), 5)` is computed alongside the existing
`rendered_vendor_count` and added to the hook key_stats. Data: two contextual
phrase replacements, identical "leading vendors" wording in both:
- description: "...and vendor-by-vendor strengths and weaknesses." -> "...and
  strength and weakness profiles for the leading vendors."
- prose: "...where each product holds strength or shows weakness." -> "...where
  the leading vendors show strength or weakness."

## Intentional

- **Honesty of claim, not raising the cap.** The catalog's other option ("ensure
  all N charted+profiled") would mean authoring strength/weakness profile PROSE
  for Zoho CRM + Pipedrive in the already-published post -- inventing content for
  vendors that weren't profiled, which the repo's "never invent content" rule
  forbids. The `[:5]` cap is a deliberate length choice; the defect is the claim,
  so the claim is what changes.
- **Left the body's "vendor-by-vendor churn risk scores".** That refers to the
  urgency chart (7 vendors), which is accurate -- not a profile claim.

## Deferred

- **D2/D3/D4** (pipedrive cluster).

## Verification

- New `test_market_landscape_exposes_capped_profile_count`: 7 profiles available
  -> hook `profile_count == 5`. Verified it FAILS on revert (`profile_count =
  len(vendor_profiles)` -> `7 == 5`).
- `pytest` generation + quote-gate suites -> 195 passed; both copies
  byte-identical.
- Data: over-claim grep -> 0 ("vendor-by-vendor strengths"/"each product holds"
  gone); "vendor-by-vendor churn risk scores" (chart) left intact; audit clean.

## Estimated diff size

| Area | LOC |
|---|---:|
| 2 generator copies (profile_count + key_stat) | ~18 |
| Test | ~30 |
| crm-landscape data (2 phrase softenings) | ~4 |
| Plan doc | ~85 |
| **Total** | **~135** |

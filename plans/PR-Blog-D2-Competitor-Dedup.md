# PR-Blog-D2-Competitor-Dedup

Ownership lane: `content-ops/blog-d2-competitor-dedup`

## Why this slice exists

Defect **D2** (`reports/blog-audit-findings.md`), first of the pipedrive cluster.
`pipedrive-deep-dive` reads "compare Pipedrive to **six** primary alternatives:
HubSpot, Salesforce, Zoho, Monday, and Zoho CRM" -- says six, lists five. The
generator's competitive-landscape section took `commonly_compared_to[:6]` with
no self-exclusion and no dedup, and the count was stated independently of the
rendered list.

DB confirms the cause is broader than the catalog noted: a vendor appears in its
OWN `commonly_compared_to` by mention count (Pipedrive self-lists at 20), and
casing variants duplicate ("HubSpot"/"Hubspot", "Zoho"/"ZOHO"). Self-inclusion
is **systemic** -- CrowdStrike, HubSpot, Mailchimp, WooCommerce, Asana, ClickUp,
and others self-list too -- so this generator fix corrects every deep-dive.

## Scope (this PR)

- **Generator** (`_blueprint_vendor_deep_dive`, both byte-identical copies):
  exclude the vendor itself and case-dedup BEFORE the `[:6]` cap, and state the
  count in `data_summary` so the description uses a deterministic number.
- **Data** (`pipedrive-deep-dive-2026-04.ts`): "six" -> "five" (the count of
  names actually listed). The name list is unchanged.

### Files touched

- `plans/PR-Blog-D2-Competitor-Dedup.md`
- `atlas_brain/autonomous/tasks/b2b_blog_post_generation.py`
- `extracted_content_pipeline/autonomous/tasks/b2b_blog_post_generation.py`
- `tests/test_b2b_blog_post_generation.py`
- `atlas-churn-ui/src/content/blog/pipedrive-deep-dive-2026-04.ts`

## Mechanism

The `comp_names = compared[:6]` comprehension is replaced with a loop that skips
`name.lower() == vendor.lower()` (self) and `name.lower() in seen` (case dup)
before appending, capping at 6 distinct external alternatives. `data_summary`
becomes "Commonly compared to {len(comp_names)} alternatives: {list}." so the
count is derived from the rendered list, not miscounted by the LLM.

## Intentional

- **Self-exclude + case-dedup only.** Both are mechanical and unambiguous.
- **"Zoho" vs "Zoho CRM" left distinct (deferred).** That is a suite-vs-product
  registry decision; `_canonicalize_competitor` (resolve_vendor_name_cached)
  returned both unchanged in an isolated import, so its production merge
  behavior is unverified -- merging product variants is not a per-PR call.

## Deferred

- **D2-followup: vendor suite/product alias unification** ("Zoho" vs "Zoho CRM"),
  a vendor-registry concern.
- **D3** (prose-vs-chart "dominant pain") and **D4** (strengths/weaknesses chart
  mislabel) -- separate slices.

## Verification

- New `test_deep_dive_competitor_list_excludes_self_and_case_dedups`: feeds
  `[Pipedrive, HubSpot, Hubspot, Salesforce, Zoho, ZOHO, Monday, Zoho CRM]` for
  vendor=Pipedrive and asserts `competitors == [HubSpot, Salesforce, Zoho,
  Monday, Zoho CRM]` (self gone, case dups gone, Zoho/Zoho CRM kept distinct)
  and "5 alternatives" in data_summary. Verified it FAILS on revert to the raw
  `compared[:6]` (got `[Pipedrive, ..., Zoho, ZOHO]`).
- `pytest` generation + quote-gate suites -> 197 passed; both copies
  byte-identical.
- Data: pipedrive post now reads "five primary alternatives: ..."; audit clean.

## Estimated diff size

| Area | LOC |
|---|---:|
| 2 generator copies (self-exclude + dedup loop + count) | ~44 |
| Test | ~45 |
| pipedrive data (one word) | ~2 |
| Plan doc | ~90 |
| **Total** | **~180** |

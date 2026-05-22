# PR-Blog-D7c-Landscape-Scope

Ownership lane: `content-ops/blog-d7c-landscape-scope`

## Why this slice exists

Defect **D7** (`reports/blog-audit-findings.md`), landscape instance. The
`crm-landscape` post claims "3,287 enriched reviews ... (172 verified + 3,115
community)" -- impossible for the CRM category (all-source enriched ceiling
1,959; allowlist enriched 1,133). PR-D7a (#760) fixed the vendor deep-dive total;
this slice fixes the category-LEVEL (landscape/roundup/best-fit) corpus counts.

Root cause (DB-verified, not the `SUM(cs.total_reviews)` theory the DB earlier
refuted at 107): a category-level topic derives its vendor set from the category,
then counts reviews by **vendor mention** with no `product_category` filter. So a
free-form community thread (Reddit etc.) that merely *mentions* a CRM vendor
among other tools is counted as a CRM "enriched review." Verified for CRM:

| | mention-scoped (buggy) | category-scoped (correct) |
|---|---:|---:|
| total enriched | ~5,931 | **1,133** |
| community split | ~5,442 | **963** |
| verified split | (≈ correct) | **170** |

Verified sources (G2/Gartner/PeerSpot) are category-aligned, so their count
(170 ≈ the post's 172) was already right -- only the community number inflates.

## Scope (this PR)

Two mention-scoped reads in the data-context builder (`_gather_data`) feed the
inflated headline, both gated to the new `category_scope` (set only for
category-level topics, where the vendor set is derived from the category):

- the `ctx_row` aggregate -> `enriched_count` / `total_reviews_analyzed`
  (the "3,287 enriched");
- `_fetch_source_distribution` -> `verified_count` / `community_count`
  (the "172 verified + 3,115 community").

Vendor-level topics (deep-dive/showdown/switching) keep vendor-mention scope --
that path is correct and PR-D7a already aligned its source allowlist.

### Files touched

- `plans/PR-Blog-D7c-Landscape-Scope.md`
- `atlas_brain/autonomous/tasks/b2b_blog_post_generation.py`
- `extracted_content_pipeline/autonomous/tasks/b2b_blog_post_generation.py`
- `tests/test_b2b_blog_post_generation.py`

## Mechanism

`_gather_data` captures `category_scope = topic_ctx["category"]` exactly where it
already fills the vendor set from the category. When set, the `ctx_row` query
scopes `WHERE product_category = $1` (instead of `EXISTS vm.vendor_name = ANY`),
and `_fetch_source_distribution` is called with `category=category_scope`.
`_fetch_source_distribution` gains a keyword `category`: when present it counts
`COUNT(DISTINCT r.id)` over `r.product_category = $1` (dropping the
vendor-mentions join); when absent it keeps the existing vendor-mention query.
Both byte-identical `b2b_blog_post_generation.py` copies (edit one, `cp` to the
mirror; verified identical).

## Intentional

- **Category scope for category-level topics only.** Gated on `category_scope`,
  so vendor deep-dives are untouched -- mention-scoping is correct for them.
- **Match D7a's scope vocabulary.** Same `_blog_source_allowlist()` + dedup;
  this slice only swaps the vendor-mention predicate for a `product_category`
  predicate on the category path.

## Deferred

- **PR-D7b (data fix):** correct the published `crm-landscape` + `zoho-crm-deep-dive`
  numbers to the category-scoped values (like #745), plus the
  `project-management-landscape` L266 orphan from #757.
- **D8** (vendor-count vs coverage) and **D2/D3/D4** (pipedrive) remain.

## Verification

- New `test_fetch_source_distribution_scopes_by_category`: with `category=`, the
  query filters `r.product_category = $1`, binds the category, and drops the
  vendor-mentions join. `test_fetch_source_distribution_reads_vendor_mentions`
  still pins the vendor-level path.
- `pytest` generation + quote-gate suites -> 192 passed; existing `_gather_data`
  tests unaffected.
- Both copies byte-identical after edit.
- Live DB: category-scoped CRM enriched = 1,133 (verified 170 / community 963)
  vs the mention-scoped ~5,931 / 5,442 the old path produced.

## Estimated diff size

| Area | LOC |
|---|---:|
| 2 generator copies (category gate + scoped queries, incl. duplicated ctx_row branch) | ~180 |
| Test | ~25 |
| Plan doc | ~90 |
| **Total** | **~295** |

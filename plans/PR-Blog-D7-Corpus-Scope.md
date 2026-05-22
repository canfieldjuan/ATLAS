# PR-Blog-D7-Corpus-Scope

Ownership lane: `content-ops/blog-d7-corpus-scope`

## Why this slice exists

Defect **D7** (`reports/blog-audit-findings.md`): a blog post's headline
"reviews collected" total is computed over a BROADER source scope than its
"enriched"/churn numbers, so the stated corpus overstates the base actually
analyzed -- and worse, "N collected / M enriched" do not share a denominator.

Verified against the live DB (Zoho CRM, all-time, deduped):

| metric | value |
|---|---:|
| `_fetch_vendor_stats.total` (all sources) | 1043 |
| same, scoped to the blog source allowlist | 441 |
| enriched (allowlist) | 263 |

The published `zoho-crm-deep-dive` headline ("~940 collected, 268 enriched")
pairs an all-source total with an allowlist-scoped enriched count -- the D7
mismatch. This slice is the generator fix for the vendor-level path; the
landscape overstatement and the published-data corrections are separate slices
(see Deferred).

## Scope (this PR)

`_fetch_vendor_stats` (both byte-identical `b2b_blog_post_generation.py` copies)
is the single source of the all-source total: its result flows into
`ctx["review_count"]` / `total_reviews` / `reviews_a/b` for the vendor_deep_dive,
vendor_showdown, switching_story, and pricing_reality_check topic types. The
sibling category path (`_fetch_category_topic_stats`) and the
`_build_blog_topic_context` ctx queries already scope to
`_blog_source_allowlist()`; only `_fetch_vendor_stats` did not.

### Files touched

- `plans/PR-Blog-D7-Corpus-Scope.md`
- `atlas_brain/autonomous/tasks/b2b_blog_post_generation.py`
- `extracted_content_pipeline/autonomous/tasks/b2b_blog_post_generation.py`
- `tests/test_b2b_blog_post_generation.py`

## Mechanism

Add `AND r.source = ANY($2)` to the `_fetch_vendor_stats` aggregate, binding
`_blog_source_allowlist()` as the second parameter -- the exact scope the
enriched/churn ctx queries already use. The function signature is unchanged
(sources are fetched inside), so no call site changes. This scopes `total`,
`enriched`, `negative`, and `avg_urgency` consistently to the allowlist, so the
headline "collected" total now shares a denominator with the enriched count
(441 / 263 for Zoho, not 1043 / 263). Applied to both byte-identical copies
(edit one, `cp` to the mirror; verified identical).

## Intentional

- **Single-point fix.** `_fetch_vendor_stats` is the only vendor-level stat
  source that omitted the allowlist; fixing it propagates to every topic type
  that reads `stats["total"]` without touching their call sites.
- **Match the existing scope, nothing more.** No `is_primary` or window filter
  added -- the confirmed mismatch is source scope alone; the enriched/churn ctx
  queries this aligns to also use source + dedup only.

## Deferred

- **PR-D7c (landscape overstatement):** `crm-landscape` shows "3,287 enriched"
  -- impossible (CRM all-source enriched ceiling is 1,959; allowlist enriched
  1,133). My first hypothesis (`read_market_landscape_candidates`'
  `SUM(cs.total_reviews)`) was REFUTED by the DB (that sum = 107), so the exact
  generator path is still being traced -- a separate slice, no guessing.
- **PR-D7b (data fix):** correct the published `zoho-crm-deep-dive` and
  `crm-landscape` numbers once the generator paths are fixed (like #745).

## Verification

- New regression test `test_fetch_vendor_stats_scopes_total_to_source_allowlist`
  fails pre-fix (query had no `r.source = ANY(`), passes after; asserts the
  bound arg equals `_blog_source_allowlist()`.
- `pytest` generation + quote-gate suites -> 185 passed.
- Both `b2b_blog_post_generation.py` copies byte-identical after edit.
- Live DB: the fixed query returns 441 (allowlist) for Zoho CRM vs 1043
  (all-source) before -- consistent with enriched 263.

## Estimated diff size

| Area | LOC |
|---|---:|
| 2 generator copies (allowlist filter + comment) | ~40 |
| Test | ~35 |
| Plan doc | ~85 |
| **Total** | **~160** |

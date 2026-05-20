# PR-Blog-GEO-Readiness-Summary

## Why this slice exists

PR-Blog-GEO-Contract-Definition defined GEO draft readiness, but generated blog
review/export rows still only show SEO/AEO readiness. Operators need to see
which generated drafts satisfy the new GEO contract before we decide which GEO
checks should block saves.

This slice adds read-only GEO readiness output to blog generated-asset rows and
CSV exports.

## Scope (this PR)

1. Add deterministic `geo_readiness` checks to blog draft export rows.
2. Add `geo_readiness` to blog CSV export columns.
3. Keep existing `output_checks` / `passed_output_checks` wired to SEO/AEO so
   no generic generated-asset UI behavior changes in this slice.
4. Add focused tests for ready and incomplete GEO readiness output.
5. Add generated-asset API coverage proving blog rows expose `geo_readiness`.

### Files touched

| File | Change |
|---|---|
| `plans/PR-Blog-GEO-Readiness-Summary.md` | Plan doc for this slice. |
| `extracted_content_pipeline/blog_post_export.py` | Add blog GEO readiness summary fields. |
| `tests/test_extracted_blog_post_export.py` | Cover ready/incomplete GEO readiness and CSV columns. |
| `tests/test_extracted_content_asset_api.py` | Prove generated-asset API rows expose GEO readiness. |

## Mechanism

The export helper inspects the generated draft content, metadata, and
data_context to derive the seven draft-level checks from the GEO contract:

- `entity_clarity`
- `answer_first_sections`
- `citable_section_structure`
- `evidence_specificity`
- `freshness_context`
- `faq_coverage`
- `citation_safety`

The row exposes a `geo_readiness` object with the same shape as the existing
`seo_aeo_readiness`: status, passed, total, missing, and checks.

## Intentional

- No save-time GEO gate. This slice adds visibility before blocking behavior.
- No publish-level GEO checks. Those belong to frontend/public-route
  verification, not blog draft export.
- No product copy changes.
- No prompt changes.
- `output_checks` remains SEO/AEO-only until the UI has a clear contract for
  showing multiple readiness groups.

## Deferred

- Decide which GEO checks should block draft save.
- Add save-time GEO gate after review output is proven useful.
- Add publish-level GEO verification for crawler-visible HTML, canonical URLs,
  BlogPosting schema, FAQ schema, breadcrumbs, OG images, and indexability.
- Consider sharing GEO helper logic with the blog quality gate once the
  review-only contract stabilizes.

## Verification

- Focused blog export and generated-asset API tests -> 8 passed.
- Python compile check over edited modules/tests -> passed.
- Diff whitespace check -> passed.
- Full extracted pipeline checks -> 1531 passed, 1 existing torch/pynvml warning.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan | ~80 |
| Blog export helper | ~150 |
| Tests | ~75 |
| **Total** | **~305** |

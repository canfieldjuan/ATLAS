# PR-Landing-Page-Readiness-Helper

## Why this slice exists

PR #709 defined the landing-page SEO/AEO/GEO contract, but generated
landing-page review rows still do not expose the readiness fields that the
review UI already knows how to render.

This slice adds draft-level readiness output at the export/API boundary. It
does not change the generator prompt, block saves, change the database schema,
or claim publish-level SEO/GEO support. Operators get a deterministic review
signal first, then later slices can decide which checks should become quality
gate blockers.

## Scope (this PR)

1. Add landing-page `seo_aeo_readiness` and `geo_readiness` summaries to draft
   export rows.
2. Expose `output_checks` and `passed_output_checks` for landing pages using
   the same generated-asset row convention as blog posts.
3. Include readiness fields in landing-page CSV exports.
4. Cover ready, incomplete, CSV, and generated-asset API visibility paths.

### Files touched

| File | Change |
|---|---|
| `plans/PR-Landing-Page-Readiness-Helper.md` | Plan doc for this implementation slice. |
| `extracted_content_pipeline/landing_page_export.py` | Add deterministic landing-page readiness helpers and export fields. |
| `tests/test_extracted_landing_page_export.py` | Cover ready/incomplete readiness output and CSV columns. |
| `tests/test_extracted_content_asset_api.py` | Prove generated-asset API rows expose landing-page readiness fields. |

## Mechanism

`_draft_row` now computes landing-page readiness summaries before returning the
review/export row.

The SEO/AEO summary checks:

- `title_tag`
- `meta_description`
- `slug_quality`
- `metadata_consistency`
- `answer_first_hero`
- `problem_solution_clarity`
- `audience_specificity`
- `objection_coverage`

The GEO summary checks:

- `offer_entity_clarity`
- `audience_entity_clarity`
- `answer_extractability`
- `section_semantics`
- `trust_signal_visibility`
- `conversion_path_clarity`
- `claim_safety`

The helper uses deterministic text, metadata, slug, section, CTA, evidence, and
placeholder checks. It treats the row as a draft review surface, so placeholder
or missing fields produce `needs_review` rather than blocking persistence.

## Intentional

- No prompt changes.
- No quality-gate blockers.
- No database schema changes.
- No frontend code changes; the review UI already has landing-page readiness
  labels/panels when rows include the fields.
- No publish-level crawler, structured-data, canonical, or rendered HTML
  verification.

## Deferred

- Extend `extracted_quality_gate/landing_page_pack.py` with selected blocking
  checks.
- Update `digest/landing_page_generation.md` so generated drafts are more
  likely to satisfy the readiness contract.
- Add public renderer/publish verification once generated landing pages have a
  concrete hosted route.

## Verification

- `pytest tests/test_extracted_landing_page_export.py tests/test_extracted_content_asset_api.py -q`
  -> passed 34/34 tests.
- Python compile command over `extracted_content_pipeline/landing_page_export.py`,
  `tests/test_extracted_landing_page_export.py`, and
  `tests/test_extracted_content_asset_api.py` -> passed 3/3 files.
- `git diff --check` -> passed with 0 whitespace errors.
- `bash scripts/local_pr_review.sh origin/main` -> passed 3/3 top-level
  checks: pre-push audit wrapper, plan/code consistency, and `git diff
  --check`.
- The pre-push audit wrapper inside local review reported all 8 internal checks
  passed: MCP tool counts, MCP port assignments, MCP tool-name inventories,
  extracted manifest sync, plan shape, plan files touched, plan diff size, and
  ASCII Python policy.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan | ~105 |
| Landing-page export helper | ~315 |
| Tests | ~200 |
| Total | ~620 |

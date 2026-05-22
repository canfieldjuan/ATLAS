# PR-Landing-Page-AEO-GEO-Section-Readiness

## Why this slice exists

PR #744 made the landing-page prompt ask for AEO/GEO section metadata:
`kind`, `primary_question`, and `answer_summary`. The review/export readiness
helpers still score only visible text, section titles, and regex matches.

This slice makes readiness use that new source metadata directly while keeping
the output contract stable.

## Scope (this PR)

1. Let problem/solution clarity read section `metadata.kind`.
2. Let objection coverage read section `metadata.kind`.
3. Let answer extractability pass from a visible first-section answer summary.
4. Let section semantics require valid section kind metadata and visible
   answer summaries for question-shaped sections.
5. Add focused export tests for the metadata scoring behavior.

### Files touched

- `plans/PR-Landing-Page-AEO-GEO-Section-Readiness.md`
- `extracted_content_pipeline/landing_page_export.py`
- `tests/test_extracted_landing_page_export.py`

## Mechanism

The export readiness helpers already own read-only `seo_aeo_readiness` and
`geo_readiness` summaries. This slice extends those helpers to inspect
`LandingPageSection.metadata`:

- `kind` must be one of the section roles requested by the prompt.
- `primary_question` marks a question-shaped section.
- `answer_summary` must be present at the start of visible `body_markdown`
  when the section is question-shaped.

The readiness output shape, check names, and totals stay the same. This is a
scoring improvement, not a new API contract.

## Intentional

- No generator prompt changes; #744 already changed the prompt.
- No parser or persistence changes; section metadata already round-trips.
- No quality-gate blocker. Older drafts can still persist; they may simply
  score lower on review readiness until regenerated or edited.
- No publish-level structured-data work.

## Deferred

- `PR-Landing-Page-Publish-Structured-Data` can map FAQ/objection sections into
  public JSON-LD once a generated landing-page renderer exists.
- `PR-Landing-Page-Section-Metadata-Quality-Gate` can decide whether missing
  or invalid section metadata should become a warning in the quality pack.

## Verification

- `pytest tests/test_extracted_landing_page_export.py tests/test_extracted_content_asset_api.py -q` - 35 passed.
- Python compile command over `extracted_content_pipeline/landing_page_export.py`
  and `tests/test_extracted_landing_page_export.py` - passed.
- `git diff --check` - passed.
- `bash scripts/local_pr_review.sh origin/main` - passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan | ~70 |
| Readiness helpers | ~90 |
| Tests | ~140 |
| **Total** | **~300** |

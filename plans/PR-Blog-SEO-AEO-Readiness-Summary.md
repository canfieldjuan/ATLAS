# PR-Blog-SEO-AEO-Readiness-Summary

## Why this slice exists

PR-Blog-SEO-Field-Persistence made generated blog SEO fields durable by writing
them into the first-class `blog_posts` SEO columns and hydrating them back into
draft metadata. Operators can now preserve the fields, but the generated-asset
review/export surface still does not tell them whether a draft has the basic
SEO/AEO pieces we are comfortable claiming.

This slice adds a deterministic readiness summary for blog drafts only. It
keeps the claim narrow: SEO metadata and answer-engine-friendly structure. GEO
remains deferred until the product contract is defined.

## Scope (this PR)

1. Add blog post SEO/AEO readiness checks to the blog draft export helper.
2. Surface `output_checks`, `passed_output_checks`, and a
   `seo_aeo_readiness` summary in blog generated-asset rows.
3. Include the readiness fields in CSV export columns.
4. Add focused tests for ready and incomplete drafts, plus generated-asset API
   visibility.

### Files touched

| File | Change |
|---|---|
| `plans/PR-Blog-SEO-AEO-Readiness-Summary.md` | Plan doc for this slice. |
| `extracted_content_pipeline/blog_post_export.py` | Add deterministic SEO/AEO readiness summary fields. |
| `tests/test_extracted_blog_post_export.py` | Cover ready/incomplete readiness output and CSV columns. |
| `tests/test_extracted_content_asset_api.py` | Prove blog generated-asset API rows expose readiness fields. |

## Mechanism

The blog export row already merges generation metadata into the review row.
This slice adds a pure helper that inspects the draft metadata and content and
returns boolean checks for:

- SEO title present and at most 60 characters.
- SEO description present and at most 155 characters.
- Target keyword present.
- At least one secondary keyword.
- At least three FAQ entries.
- Question-style H2 heading or answer-first section structure detectable in the
  Markdown content.

The row exposes those booleans as `output_checks`, counts passed checks in
`passed_output_checks`, and adds `seo_aeo_readiness` with total, passed,
missing, and status fields. The title/description checks use `*_ready` names
because they validate both presence and length.

## Intentional

- No prompt changes. This slice measures the current output contract instead of
  changing generation behavior.
- No GEO score. GEO needs a definition before it can become a validator.
- No frontend UI code. The generated-asset UI already has generic support for
  `output_checks` / `passed_output_checks`; this slice focuses on the backend
  row shape.
- The AEO structure check is intentionally heuristic. It detects obvious
  question-style H2s or answer-first section openings; it is not a full content
  quality gate.

## Deferred

- Add a dedicated SEO/AEO/GEO quality gate before draft save.
- Add a first-class GEO readiness definition.
- Add richer frontend labels for blog readiness if the generic generated-asset
  display is not enough.
- Audit public publish pages end to end after readiness output lands.

## Verification

- Focused blog export and generated-asset API tests -> 7 passed.
- Python compile check over edited modules/tests -> passed.
- Diff whitespace check -> passed.
- Extracted content pipeline validation -> passed.
- Atlas reasoning import guard for extracted_content_pipeline -> passed.
- Standalone extracted audit with fail-on-debt -> passed.
- ASCII Python policy check -> passed.
- Full extracted pipeline checks -> 1527 passed, 1 existing torch/pynvml warning.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan | ~85 |
| Blog export helper | ~85 |
| Tests | ~95 |
| **Total** | **~270** |

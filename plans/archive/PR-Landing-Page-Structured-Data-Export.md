# PR-Landing-Page-Structured-Data-Export

## Why this slice exists

Landing-page generation now produces SEO/AEO/GEO section metadata and the
export path scores readiness, but hosts still do not get renderer-ready
structured data back from the generated asset export. There is no clear public
generated landing-page renderer in this repo yet, so publishing JSON-LD inside
the export contract is the next production step before wiring a live route.

## Scope (this PR)

Ownership lane: content-ops/landing-page-structured-data-export

1. Add a reusable landing-page Schema.org structured-data builder.
2. Export structured data alongside existing SEO/AEO and GEO readiness fields.
3. Type the new export field in the Intel UI content-assets API model.
4. Cover WebPage and FAQPage output with focused tests.
5. Keep the builder conservative: no invented public URL, organization,
   pricing, or proof claim.

### Files touched

- `plans/PR-Landing-Page-Structured-Data-Export.md`
- `extracted_content_pipeline/manifest.json`
- `extracted_content_pipeline/landing_page_export.py`
- `extracted_content_pipeline/landing_page_structured_data.py`
- `atlas-intel-ui/src/api/contentOps.ts`
- `tests/test_landing_page_structured_data.py`
- `tests/test_extracted_landing_page_export.py`

## Mechanism

The structured-data builder returns a JSON-LD object with
`@context: https://schema.org` and an `@graph`.

The builder always emits a `WebPage` node from the draft title, meta
description, persona, value proposition, CTA, and ordered sections. It adds
`@id` / `url` only when the draft meta already provides `canonical_url`,
`public_url`, or `url`.

The builder emits an additional `FAQPage` node only from section metadata where
`primary_question` and a visible `answer_summary` are present. This keeps the
published structured data aligned with the same answer-first metadata contract
used by readiness scoring.

## Intentional

- No live route or renderer changes in this PR.
- No database migration. The structured data is derived at export time from
  existing draft fields.
- No generated organization, offer, pricing, aggregate rating, or proof schema.
  Those require source data that is not guaranteed in a landing-page draft.

## Deferred

- `PR-Landing-Page-Public-Renderer` can consume the exported structured data
  once a public generated landing-page route exists.
- `PR-Landing-Page-Structured-Data-Preview-UI` can add an explicit review panel
  for JSON-LD inspection if the asset review screen needs it.

## Verification

- `pytest tests/test_landing_page_structured_data.py tests/test_extracted_landing_page_export.py -q` - 12 passed.
- Python compile over `extracted_content_pipeline/landing_page_structured_data.py` and `extracted_content_pipeline/landing_page_export.py` - passed.
- `scripts/run_extracted_pipeline_checks.sh` - 1628 passed.
- `npm run build` in `atlas-intel-ui` - passed.
- `git diff --check` - passed.
- `bash scripts/local_pr_review.sh origin/main` - passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan | ~65 |
| Structured-data builder | ~170 |
| Export/API wiring | ~15 |
| Tests | ~170 |
| **Total** | **~420** |

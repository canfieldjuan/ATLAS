# PR: Generated Asset API Draft Id Filters

## Why this slice exists

The support-ticket live smoke prints exact saved draft ids. The export CLI can
use those ids after the previous slice, but the hosted generated-assets API
still only supports broad filters such as topic type, campaign name, and slug.
The API review path should inspect the same exact generated row the CLI can.

## Scope (this PR)

Ownership lane: content-ops/support-ticket-input-provider

Slice phase: Functional validation

1. Add repeatable `id` query filters to generated-asset draft list and export
   endpoints.
2. Thread ids through the existing generated-assets API export helper for
   `blog_post` and `landing_page`.
3. Reject `id` filters for unsupported asset types with a clear 400 response.
4. Add focused API tests proving query binding and unsupported-asset behavior.

### Files touched

- `plans/PR-Generated-Asset-API-Draft-Id-Filters.md`
- `extracted_content_pipeline/api/generated_assets.py`
- `tests/test_extracted_content_asset_api.py`

## Mechanism

FastAPI receives `?id=<draft-id>` as a repeatable query alias and passes the
normalized ids into `_export_for_asset`. Blog-post and landing-page exports use
the repository id filters introduced by the prior CLI slice. Other assets fail
fast because their export helpers do not accept exact-id filters yet.

## Intentional

- API parity only. No generation, review-status, file-ingestion, or FAQ changes.
- Exact id filtering remains tenant-scoped by the existing repositories.
- This is stacked on the draft-id export helper slice until that PR lands.

## Deferred

- Report, sales brief, and FAQ Markdown exact-id export can be added later if
  those lanes need it.
- Parked hardening: none. `HARDENING.md` was scanned; current entries are FAQ
  scale and file-ingestion concurrency work outside this support-ticket provider
  validation slice.

## Verification

- Generated asset API tests:
  `pytest tests/test_extracted_content_asset_api.py -q`
  - 59 passed.
- Py compile for changed Python files - passed.
- Git whitespace check - passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan | ~55 |
| API | ~20 |
| Tests | ~35 |
| **Total** | **~110** |

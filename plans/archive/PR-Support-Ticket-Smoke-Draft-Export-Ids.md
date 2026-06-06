# PR: Support Ticket Smoke Draft Export Ids

## Why this slice exists

The support-ticket provider live smoke now saves real landing-page and blog-post
drafts, but inspecting the exact generated draft still requires broad export
filters or manual DB lookup. That slows validation of the actual generated
content and makes it easier to inspect the wrong row when multiple smoke runs
share an account.

## Scope (this PR)

Ownership lane: content-ops/support-ticket-input-provider

Slice phase: Functional validation

1. Add exact draft-id filtering to landing-page and blog-post draft export
   helpers.
2. Add `--id` to `scripts/export_extracted_content_assets.py` for those assets.
3. Document the smoke follow-up command that exports the saved draft by id.
4. Add focused tests for helper forwarding and CLI SQL binding.

### Files touched

- `plans/PR-Support-Ticket-Smoke-Draft-Export-Ids.md`
- `extracted_content_pipeline/README.md`
- `extracted_content_pipeline/blog_ports.py`
- `extracted_content_pipeline/blog_post_export.py`
- `extracted_content_pipeline/blog_post_postgres.py`
- `extracted_content_pipeline/landing_page_export.py`
- `extracted_content_pipeline/landing_page_ports.py`
- `extracted_content_pipeline/landing_page_postgres.py`
- `scripts/export_extracted_content_assets.py`
- `tests/test_extracted_blog_post_export.py`
- `tests/test_extracted_content_asset_export_cli.py`
- `tests/test_extracted_landing_page_export.py`

## Mechanism

- Thread optional `ids` through `export_landing_page_drafts` and
  `export_blog_post_drafts`.
- Add optional `ids` to the Postgres repository `list_drafts` methods and use
  tenant-scoped `id = ANY(...)` filters.
- Keep existing status, topic, campaign, slug, and limit filters intact.

## Intentional

- This only changes read-only export/inspection paths.
- No production generation, FAQ generator, file-ingestion, or review-status
  behavior changes.
- Exact ids remain tenant-scoped through the existing repository `scope`.

## Deferred

- The live smoke script still writes the execution JSON separately; this slice
  documents the export command rather than coupling export side effects into the
  smoke runner.
- Parked hardening: none. `HARDENING.md` was scanned; the current entries are
  FAQ scale and file-ingestion concurrency work outside this support-ticket
  provider validation slice.

## Verification

- Focused export tests:
  `pytest tests/test_extracted_blog_post_export.py tests/test_extracted_landing_page_export.py tests/test_extracted_content_asset_export_cli.py -q`
  - 38 passed.
- Py compile for changed Python files - passed.
- Git whitespace check - passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan | ~70 |
| Export helpers and repository filters | ~45 |
| CLI | ~15 |
| Tests | ~45 |
| Docs | ~15 |
| **Total** | **~190** |

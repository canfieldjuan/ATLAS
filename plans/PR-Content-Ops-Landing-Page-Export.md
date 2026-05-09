# PR-Content-Ops-Landing-Page-Export

## Why this slice exists

Campaign drafts and structured reports now have read-only JSON/CSV export
helpers for host review. Landing pages have a standalone repository and
`list_drafts()` method, but no matching export seam.

## Scope (this PR)

1. Add a landing-page-only export helper over `LandingPageRepository.list_drafts()`.
2. Support JSON and CSV output from exported landing page rows.
3. Preserve page metadata and expose generation/reasoning summary fields.
4. Persist compact reasoning metadata from generated landing pages so export
   rows reflect real generated drafts, not only hand-seeded metadata.
5. Document the new helper in package status/productization docs.

### Files touched

- `extracted_content_pipeline/landing_page_export.py`
- `extracted_content_pipeline/landing_page_generation.py`
- `tests/test_extracted_landing_page_export.py`
- `extracted_content_pipeline/README.md`
- `extracted_content_pipeline/STATUS.md`
- `extracted_content_pipeline/docs/standalone_productization.md`
- `extracted_content_pipeline/manifest.json`
- `scripts/run_extracted_pipeline_checks.sh`
- `docs/extraction/coordination/inflight.md`

## Mechanism

`export_landing_page_drafts(repository, scope=..., ...)` calls the host-provided
`LandingPageRepository.list_drafts()` and converts each `LandingPageDraft` to a
serializable row. The result mirrors the existing report export ergonomics:

- `as_dict()` for JSON responses or host scripts.
- `as_csv()` for review spreadsheets.
- `filters` and `limit` metadata on the export result.

## Intentional

- No API route or CLI in this first landing-page export slice.
- No shared generic asset-export abstraction yet. Landing pages, reports, and
  sales briefs have different review columns.

## Deferred

- Sales brief export helper.
- Host-mounted API routes / CLIs for landing page exports.

## Verification

Completed:

- `python -m pytest tests/test_extracted_landing_page_export.py tests/test_extracted_landing_page_generation.py`
- `python -m py_compile extracted_content_pipeline/landing_page_export.py extracted_content_pipeline/landing_page_generation.py tests/test_extracted_landing_page_export.py`
- `bash scripts/run_extracted_pipeline_checks.sh`
- `git diff --check`
- Non-ASCII byte check for edited Python files.

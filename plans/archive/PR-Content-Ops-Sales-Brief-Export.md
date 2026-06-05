# PR-Content-Ops-Sales-Brief-Export

## Why this slice exists

Campaigns, reports, and landing pages now have read-only JSON/CSV export
helpers for host review. Sales briefs have a standalone repository and
`list_drafts()` method, but no matching export seam.

## Scope (this PR)

1. Add a sales-brief-only export helper over `SalesBriefRepository.list_drafts()`.
2. Support JSON and CSV output from exported sales brief rows.
3. Preserve brief metadata and expose generation/reasoning summary fields.
4. Persist compact reasoning metadata from generated sales briefs so export
   rows reflect real generated drafts.
5. Document the new helper in package status/productization docs.

### Files touched

- `extracted_content_pipeline/sales_brief_export.py`
- `extracted_content_pipeline/sales_brief_generation.py`
- `tests/test_extracted_sales_brief_export.py`
- `extracted_content_pipeline/README.md`
- `extracted_content_pipeline/STATUS.md`
- `extracted_content_pipeline/docs/standalone_productization.md`
- `extracted_content_pipeline/manifest.json`
- `scripts/run_extracted_pipeline_checks.sh`
- `docs/extraction/coordination/inflight.md`

## Mechanism

`export_sales_brief_drafts(repository, scope=..., ...)` calls the host-provided
`SalesBriefRepository.list_drafts()` and converts each `SalesBriefDraft` to a
serializable row. The result mirrors the report and landing page export
ergonomics:

- `as_dict()` for JSON responses or host scripts.
- `as_csv()` for review spreadsheets.
- `filters` and `limit` metadata on the export result.

## Intentional

- No API route or CLI in this first sales brief export slice.
- No shared generic asset-export abstraction yet.

## Deferred

- Host-mounted API routes / CLIs for generated asset exports.

## Verification

Completed:

- `python -m pytest tests/test_extracted_sales_brief_export.py tests/test_extracted_sales_brief_generation.py`
- `python -m py_compile extracted_content_pipeline/sales_brief_export.py extracted_content_pipeline/sales_brief_generation.py tests/test_extracted_sales_brief_export.py`
- `bash scripts/run_extracted_pipeline_checks.sh`
- `git diff --check`
- Non-ASCII byte check for edited Python files.

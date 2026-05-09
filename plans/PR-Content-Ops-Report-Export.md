# PR-Content-Ops-Report-Export

## Why this slice exists

Campaign drafts have a read-only JSON/CSV export helper for host review. Reports
already have a standalone Postgres repository and `list_drafts()` method, but no
matching export seam. This is the first generated-asset export slice outside
campaign drafts.

## Scope (this PR)

1. Add a report-only export helper over `ReportRepository.list_drafts()`.
2. Support JSON and CSV output from exported report rows.
3. Preserve report metadata and expose the same generation/reasoning summary
   fields used by campaign draft exports.
4. Document the new helper in package status/productization docs.

### Files touched

- `extracted_content_pipeline/report_export.py`
- `tests/test_extracted_report_export.py`
- `extracted_content_pipeline/README.md`
- `extracted_content_pipeline/STATUS.md`
- `extracted_content_pipeline/docs/standalone_productization.md`
- `extracted_content_pipeline/manifest.json`
- `scripts/run_extracted_pipeline_checks.sh`
- `docs/extraction/coordination/inflight.md`

## Mechanism

`export_report_drafts(repository, scope=..., ...)` calls the host-provided
`ReportRepository.list_drafts()` and converts each `ReportDraft` to a
serializable row. The export result mirrors campaign export ergonomics:

- `as_dict()` for JSON responses or host scripts.
- `as_csv()` for review spreadsheets.
- `filters` and `limit` metadata on the export result.

## Intentional

- No API route or CLI in this first report-export slice; hosts can call the
  helper directly, and routes can be added once the report helper shape is
  reviewed.
- No shared generic asset-export abstraction yet. Reports, landing pages, and
  sales briefs have different row shapes; premature abstraction would obscure
  the host-facing columns.

## Deferred

- Landing page and sales brief export helpers.
- Host-mounted API routes / CLIs for report exports.

## Verification

Completed:

- `python -m pytest tests/test_extracted_report_export.py`
- `python -m py_compile extracted_content_pipeline/report_export.py tests/test_extracted_report_export.py`
- `bash scripts/run_extracted_pipeline_checks.sh`
- `git diff --check`
- Non-ASCII byte check for edited Python files.

## Estimated diff size

About 6 files, under 220 changed lines.

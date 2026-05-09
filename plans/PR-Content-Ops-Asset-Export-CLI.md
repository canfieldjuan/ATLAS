# PR-Content-Ops-Asset-Export-CLI

## Why this slice exists

Reports, landing pages, and sales briefs now have read-only export helpers, but
hosts still need to write Python to use them. Campaign drafts already have a
host-facing export CLI. Generated Content Ops assets should have the same
operator path.

## Scope (this PR)

1. Add a read-only CLI for generated asset exports:
   `scripts/export_extracted_content_assets.py`.
2. Support `--asset report|landing_page|sales_brief`.
3. Reuse the existing Postgres repository adapters and export helpers.
4. Support JSON/CSV output, tenant scope, status, limit, and asset-specific
   filters.
5. Document the command and add focused tests to extracted pipeline checks.

### Files touched

- `scripts/export_extracted_content_assets.py`
- `tests/test_extracted_content_asset_export_cli.py`
- `scripts/run_extracted_pipeline_checks.sh`
- `extracted_content_pipeline/README.md`
- `extracted_content_pipeline/STATUS.md`
- `extracted_content_pipeline/docs/standalone_productization.md`
- `docs/extraction/coordination/inflight.md`

## Intentional

- No FastAPI route in this slice.
- No custom SQL in the CLI. It constructs repositories and calls export helpers.
- One CLI for generated assets rather than three near-identical scripts.

## Verification

Completed:

- `python -m pytest tests/test_extracted_content_asset_export_cli.py`
- `python -m py_compile scripts/export_extracted_content_assets.py tests/test_extracted_content_asset_export_cli.py`
- `bash scripts/run_extracted_pipeline_checks.sh`
- `git diff --check`
- Non-ASCII byte check for edited Python files.

# PR-Content-Ops-Asset-Review-CLI

## Why this slice exists

Reports, landing pages, and sales briefs now have export helpers and a shared
export CLI. Hosts can review generated assets without SQL, but they still need
to write Python or SQL to move an exported asset to `approved`, `queued`,
`rejected`, or another lifecycle status.

## Scope (this PR)

1. Add `scripts/review_extracted_content_assets.py`.
2. Support `--asset report|landing_page|sales_brief`.
3. Reuse the existing Postgres repository adapters and their scoped
   `update_status()` methods.
4. Emit JSON for both hit and miss cases so operator automation can consume the
   result.
5. Document the command and add focused tests to extracted pipeline checks.

### Files touched

- `scripts/review_extracted_content_assets.py`
- `tests/test_extracted_content_asset_review_cli.py`
- `scripts/run_extracted_pipeline_checks.sh`
- `extracted_content_pipeline/README.md`
- `extracted_content_pipeline/STATUS.md`
- `extracted_content_pipeline/docs/standalone_productization.md`
- `docs/extraction/coordination/inflight.md`

## Intentional

- No batch update helper in this slice. Generated asset repositories expose a
  simple single-row hit/miss contract today.
- No FastAPI route in this slice.
- One CLI for generated assets rather than three near-identical scripts.

## Verification

Planned:

- `python -m pytest tests/test_extracted_content_asset_review_cli.py`
- `python -m py_compile scripts/review_extracted_content_assets.py tests/test_extracted_content_asset_review_cli.py`
- `bash scripts/run_extracted_pipeline_checks.sh`
- `git diff --check`
- Non-ASCII byte check for edited Python files.

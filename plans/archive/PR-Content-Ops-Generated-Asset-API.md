# PR-Content-Ops-Generated-Asset-API

## Why this slice exists

Campaign drafts have a host-mounted FastAPI review/export router. Generated
reports, landing pages, and sales briefs now have Postgres repositories plus
export and review CLIs, but host FastAPI installs still need custom route code
to expose the same workflow.

## Scope (this PR)

1. Add `extracted_content_pipeline/api/generated_assets.py`.
2. Support generated asset list/export/review routes for `report`,
   `landing_page`, and `sales_brief`.
3. Reuse existing Postgres repositories, export helpers, and scoped
   `update_status()` methods.
4. Preserve host-defined status strings for review updates.
5. Add focused FastAPI tests and wire them into extracted pipeline checks.

### Files touched

- `extracted_content_pipeline/api/generated_assets.py`
- `tests/test_extracted_content_asset_api.py`
- `scripts/run_extracted_pipeline_checks.sh`
- `extracted_content_pipeline/README.md`
- `extracted_content_pipeline/STATUS.md`
- `extracted_content_pipeline/docs/standalone_productization.md`
- `docs/extraction/coordination/inflight.md`

## Intentional

- No new generation or persistence path.
- No batch review endpoint in this slice.
- No Atlas API globals. Hosts inject pool and tenant scope providers.

## Deferred

- Batch review/status updates for generated assets.
- Generated asset execution routes. This router only exposes post-generation
  review/export workflows.
- Per-host status vocabulary enforcement.

## Verification

Planned:

- `python -m pytest tests/test_extracted_content_asset_api.py`
- `python -m py_compile extracted_content_pipeline/api/generated_assets.py tests/test_extracted_content_asset_api.py`
- `bash scripts/run_extracted_pipeline_checks.sh`
- `git diff --check`
- Non-ASCII byte check for edited Python files.

## Estimated diff size

- Production: ~250 LOC.
- Tests: ~250 LOC.
- Docs/check wiring: ~40 LOC.

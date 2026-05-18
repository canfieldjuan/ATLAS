# PR: Content Ops Ingestion Import API

## Why this slice exists

Hosted Content Ops can inspect pasted opportunity/source rows, but operators
still need the CLI to write accepted rows into `campaign_opportunities`.
This slice adds the hosted write seam immediately after diagnostics, reusing
the existing importer instead of creating a second ingestion path.

## Scope (this PR)

Add a host-wired `POST /content-ops/ingestion/import` route to the Content Ops
control-surface router and document the route.

### Files touched

- `extracted_content_pipeline/api/control_surfaces.py`
- `atlas_brain/api/__init__.py`
- `tests/test_extracted_content_control_surface_api.py`
- `extracted_content_pipeline/docs/control_surface_preview_api.md`
- `extracted_content_pipeline/STATUS.md`
- `docs/extraction/coordination/inflight.md`
- `docs/extraction/coordination/state.md`
- `plans/PR-Content-Ops-Ingestion-Import-API.md`

## Mechanism

- Add an optional pool provider to `create_content_ops_control_surface_router`.
- Add a bounded import payload model that extends the inspect payload with
  `replace_existing` and `dry_run`.
- Run `inspect_ingestion_rows(...)` first, fail closed when diagnostics are not
  import-ready, then call `import_campaign_opportunities(...)` with
  `normalize=False` so the inspected rows are the rows written.
- Wrap writes in a transaction when the host pool/connection supports one, and
  map unexpected database failures to a safe 503 response.
- Resolve tenant scope through the existing `scope_provider` seam.

## Intentional

- No frontend import button in this PR.
- No new table schema.
- No generated-asset execution changes.
- Dry-run imports do not require a pool provider.

## Deferred

- New-run UI import action.
- CSV/file upload import from the browser.
- Import history/audit UI.

## Verification

- Run the focused `tests/test_extracted_content_control_surface_api.py` pytest file.
- Compile `extracted_content_pipeline/api/control_surfaces.py`,
  `atlas_brain/api/__init__.py`, and
  `tests/test_extracted_content_control_surface_api.py`.
- `git diff --check`
- Run `scripts/local_pr_review.sh` from the repo root.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Control-surface import route | ~195 |
| Atlas route mount wiring | ~1 |
| API tests | ~165 |
| Docs/status/coordination | ~35 |
| Plan doc | ~60 |
| **Total** | ~456 |

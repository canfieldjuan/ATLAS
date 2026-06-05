# Content Ops Ingestion Inspect API

## Why this slice exists

PR #578 made Content Ops ingestion readiness inspectable from a host CLI. The
next practical seam is letting a hosted operator UI inspect inline rows with the
same diagnostics contract before import, generation, or database writes.

## Scope (this PR)

1. Add an in-memory ingestion diagnostics helper over existing opportunity and
   source-row normalizers.
2. Add `POST /content-ops/ingestion/inspect` to the existing control-surface
   router.
3. Add focused API tests for source rows, opportunity rows, and request bounds.
4. Document the route and update product/coordination state.

### Files touched

- `extracted_content_pipeline/ingestion_diagnostics.py`
- `extracted_content_pipeline/api/control_surfaces.py`
- `tests/test_extracted_content_control_surface_api.py`
- `extracted_content_pipeline/docs/control_surface_preview_api.md`
- `extracted_content_pipeline/STATUS.md`
- `docs/extraction/coordination/inflight.md`
- `docs/extraction/coordination/state.md`
- `plans/PR-Content-Ops-Ingestion-Inspect-API.md`

## Mechanism

`inspect_ingestion_rows(...)` reuses `normalize_campaign_opportunity_rows(...)`
and `source_rows_to_campaign_opportunities(...)`, then builds the same
`IngestionDiagnosticsReport` used by the file-backed CLI. The API route only
validates request size and delegates to that helper.

## Intentional

- No file upload or multipart handling.
- No database writes.
- No LLM calls.
- No new source aliases or source-type rules.
- No UI code.

## Deferred

- Real upload storage/import APIs remain separate.
- Rich operator-console rendering is still a frontend task.
- Additional source adapters remain blocked on a real host export fixture.

## Verification

- Run focused control-surface API tests.
- Run focused ingestion diagnostics tests.
- Compile touched Python files.
- Run the local PR review script.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Diagnostics helper | ~35 |
| API route/model | ~55 |
| Tests | ~65 |
| Docs/status/coordination | ~70 |
| **Total** | ~225 |

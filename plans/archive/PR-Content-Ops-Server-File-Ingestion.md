# PR-Content-Ops-Server-File-Ingestion

## Why this slice exists

FAQ generation can process thousands of ticket rows, but the hosted
control-surface ingestion routes still accept inline JSON row arrays capped at
1,000 rows. That makes the browser/API shape the production bottleneck for
larger support-ticket CSV exports even though the underlying generator already
handles larger files.

This slice adds a production-shaped server-side file ingestion path so larger
CSV/JSON/JSONL uploads are parsed by the API process instead of being expanded
into a giant browser JSON body. The old inline row endpoints stay working but
are marked deprecated so they can become dead code in a later cleanup.

## Scope (this PR)

Ownership lane: content-ops/server-file-ingestion

1. Add `/content-ops/ingestion/files/inspect` for uploaded ingestion files.
2. Add `/content-ops/ingestion/files/import` for uploaded ingestion files that
   reuse the existing diagnostics/import machinery.
3. Bound file uploads by size and normalized row count separately from the old
   inline-row cap.
4. Mark `/content-ops/ingestion/inspect` and `/content-ops/ingestion/import`
   as deprecated in FastAPI metadata.
5. Add focused route tests proving large files bypass the 1,000 inline-row cap
   without removing compatibility.

### Files touched

- `extracted_content_pipeline/api/control_surfaces.py`
- `tests/test_extracted_content_control_surface_api.py`
- `plans/PR-Content-Ops-Server-File-Ingestion.md`

## Mechanism

The new file routes accept multipart file uploads plus form fields mirroring
the inline routes: `source_rows`, `source`, `target_mode`,
`max_source_text_chars`, `sample_limit`, `default_fields`, and import-only
`replace_existing` / `dry_run`.

The API writes the uploaded bytes to a temporary file, calls the existing
`inspect_ingestion_file` parser/diagnostics path, then deletes the temp file.
Import routes reuse the existing `_import_campaign_opportunities_for_route`
function so scope checks, dry-run behavior, replace-existing behavior, and DB
error wrapping stay centralized.

## Intentional

- The old inline endpoints remain available because the current UI still calls
  them. This PR deprecates them in API metadata instead of removing them.
- The new file upload row cap is intentionally larger than the inline cap. The
  inline cap protects request-body shape; the file route is the production
  path for larger customer exports.
- This slice does not add durable upload/job storage. It proves the server-side
  parsing/import path first; queued jobs and Blob-backed persistence are the
  next hardening layer.

## Deferred

- Future PR: move the Intel UI file loader from inline JSON posts to the new
  file endpoints.
- Future PR: add durable upload/job persistence, polling status, and cleanup
  metadata for long-running customer CSV imports.
- Future PR: remove the deprecated inline ingestion routes after UI migration
  and compatibility window.
- Parked hardening: none.

## Verification

- `python -m pytest tests/test_extracted_content_control_surface_api.py -q`
  - `84 passed in 2.85s`
- Ran the extracted pipeline check script at
  `scripts/run_extracted_pipeline_checks.sh` with bash.
  - `1858 passed, 1 skipped, 1 warning in 19.02s`
  - All extracted content pipeline checks completed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan doc | ~85 |
| API file routes and helpers | ~230 |
| Backend API tests | ~220 |
| **Total** | ~535 |

Slightly over the 400 LOC target because the slice needs route implementation,
load-bearing upload bounds, deprecation coverage, >1,000-row file-route tests,
and rejection-path coverage in one coherent API contract change.

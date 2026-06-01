# PR: Content Ops Upload Source Run Handoff

## Why this slice exists

The landing-page and generated-blog product path is now reviewable and public,
but the New Run UI still leaves a gap between uploaded customer exports and the
generation request. Operators can load/inspect/import a CSV, and the backend can
normalize it, but execute still consumes `inputs.source_material`; the imported
target IDs shown in the UI are not automatically usable by landing/blog
generation. That makes the live path easy to false-green: the CSV import can
succeed while the later page/blog run omits the uploaded data.

This slice closes the smallest end-to-end handoff: the ingestion API can
optionally return the full normalized source material, and the UI can apply that
material to `inputs.source_material` after a successful upload/import before
preview/plan/execute.

## Scope (this PR)

Ownership lane: content-ops/upload-source-run-handoff

Slice phase: Vertical slice

1. Add an opt-in `include_source_material` flag to ingestion inspect/import
   requests, including multipart file uploads.
2. When requested, return the bounded normalized opportunities as
   `source_material` in ingestion diagnostics.
3. Thread the optional field through the frontend API/domain adapters.
4. In the New Run UI, request source material during import and provide a
   guarded apply action that writes it into `inputs.source_material`.
5. Add focused API/domain/UI source tests proving upload/import can feed the
   generation request without using samples as a lossy substitute.

### Files touched

- `extracted_content_pipeline/api/control_surfaces.py`
- `tests/test_extracted_content_control_surface_api.py`
- `atlas-intel-ui/src/api/contentOps.ts`
- `atlas-intel-ui/src/domain/contentOps/types.ts`
- `atlas-intel-ui/src/domain/contentOps/fromWire.ts`
- `atlas-intel-ui/src/pages/ContentOpsNewRun.tsx`
- `atlas-intel-ui/scripts/content-ops-ingestion-routing.test.mjs`
- `atlas-intel-ui/scripts/content-ops-upload-source-run-handoff.test.mjs`
- `atlas-intel-ui/package.json`
- `plans/PR-Content-Ops-Upload-Source-Run-Handoff.md`

## Mechanism

`ContentOpsIngestionInspectModel` gains an optional boolean
`include_source_material`, mirrored by file-upload form fields. The default is
false, so ordinary inspect/import responses remain compact. When true,
`_ingestion_diagnostics_response(...)` adds:

```json
{
  "source_material": [{ "target_id": "ticket-1", "...": "..." }]
}
```

The UI requests this on import, maps it through `fromWireIngestionDiagnostics`,
and exposes an apply control on the successful import card. Applying the import
updates `inputsJson` with `source_material` from the normalized response and
marks preview/plan/execute stale. If the response omits source material or the
current inputs JSON is invalid, the apply action fails closed with an inline
message instead of mutating the request.

## Intentional

- No persisted-target execute path in this slice. Loading imported target IDs
  back from Postgres into the provider is a larger backend orchestration slice;
  this PR proves the UI upload to generation handoff without adding another DB
  lookup surface.
- The full source material is opt-in. Existing inspect/import callers keep the
  compact diagnostics envelope.
- The UI applies `source_material`, not `samples`; samples are capped and are
  only for diagnostics.
- No live LLM, database, or hosted browser run in CI. Tests use route/unit
  fixtures and source-level UI checks.

## Deferred

- Future PR: persisted import target selection for execute, where imported
  target IDs are read back from `campaign_opportunities` by tenant scope.
- Future PR: browser E2E against a live uploaded support-ticket CSV producing
  an approved public landing/blog asset.
- Parked hardening: none. `HARDENING.md` and `ATLAS-HARDENING.md` were scanned;
  existing entries are blog content-quality issues, not this upload handoff.

## Verification

- `python -m py_compile extracted_content_pipeline/api/control_surfaces.py
  tests/test_extracted_content_control_surface_api.py` - passed.
- `pytest tests/test_extracted_content_control_surface_api.py -k
  'ingestion_inspect_route_can_include_full_source_material or
  ingestion_file_inspect_route_can_include_full_source_material or
  ingestion_file_import_route_dry_run_uses_file_parser or
  ingestion_inspect_route_reports_source_rows or
  ingestion_file_inspect_route_accepts_more_than_inline_row_cap' -q` - 5
  passed, 121 deselected.
- `pytest tests/test_extracted_content_control_surface_api.py -q` - 125
  passed, 1 skipped.
- `cd atlas-intel-ui && npm run test:content-ops-upload-source-run-handoff` -
  3 passed.
- `cd atlas-intel-ui && npm run test:content-ops-ingestion-routing` - 4
  passed.
- `cd atlas-intel-ui && npm run test:content-ops-landing-page-e2e-ui` - 4
  passed.
- `cd atlas-intel-ui && npm run build` - passed; generated 17 sitemap URLs and
  prerendered 16 public routes.
- `bash scripts/validate_extracted_content_pipeline.sh` - passed.
- `python extracted/_shared/scripts/forbid_atlas_reasoning_imports.py
  extracted_content_pipeline` - passed.
- `python scripts/audit_extracted_standalone.py --fail-on-debt` - passed.
- `bash scripts/check_ascii_python.sh` - passed.
- `bash scripts/run_extracted_pipeline_checks.sh` - passed; extracted content
  pytest reported 2879 passed, 10 skipped.
- `bash scripts/local_pr_review.sh --current-pr-body-file
  /tmp/content-ops-upload-source-run-handoff-pr-body.md` - passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan | ~130 |
| Backend API + tests | ~120 |
| Frontend API/domain | ~60 |
| UI handoff + tests | ~150 |
| **Total** | **~465** |

This is slightly over the 400 LOC soft budget because the backend opt-in
envelope, frontend wire mapping, and UI apply behavior need to ship together to
prove the uploaded CSV reaches the run request.

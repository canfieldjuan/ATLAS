# PR-Content-Ops-Ingestion-Limits-Catalog

## Why this slice exists

The Content Ops API now has two ingestion paths: multipart uploaded files for
customer CSV/JSON/JSONL exports and deprecated inline JSON rows for pasted or
manual rows. The backend enforces separate limits for each path, but
`GET /content-ops/control-surfaces` only exposes `ingestion_profiles`.

This leaves the UI and docs without a machine-readable source of truth for file
size, row caps, sample limits, and inline deprecation status. This slice adds a
small catalog contract for those limits.

## Scope (this PR)

Ownership lane: content-ops/ingestion-limits-catalog

1. Add `ingestion_limits` to the control-surface catalog response.
2. Include inline-row limits, file-upload limits, shared text/sample caps, and
   supported upload formats.
3. Thread the new wire field through frontend API/domain types and fixtures.
4. Add focused backend and frontend contract coverage.

### Files touched

- `plans/PR-Content-Ops-Ingestion-Limits-Catalog.md`
- `extracted_content_pipeline/api/control_surfaces.py`
- `tests/test_extracted_content_control_surface_api.py`
- `atlas-intel-ui/src/api/contentOps.ts`
- `atlas-intel-ui/src/domain/contentOps/types.ts`
- `atlas-intel-ui/src/domain/contentOps/fromWire.ts`
- `atlas-intel-ui/src/api/__fixtures__/contentOps/catalog.json`
- `docs/frontend/content_ops_frontend_contract.md`

## Mechanism

The static catalog payload now includes immutable ingestion-limit metadata:

```json
{
  "ingestion_limits": {
    "inline_rows": { "max_rows": 1000, "deprecated": true },
    "file_upload": {
      "max_file_bytes": 26214400,
      "max_rows": 10000,
      "supported_formats": ["auto", "json", "jsonl", "csv"]
    },
    "max_source_text_chars": 10000,
    "max_sample_limit": 25
  }
}
```

`_compose_describe_response` reprojects this metadata into fresh dict/list
objects per request, preserving the static-cache non-aliasing contract.

## Intentional

- No UI rendering in this slice. The UI can consume the new domain field in a
  follow-up without hardcoding caps.
- No route behavior changes. This only exposes limits that already exist.
- The deprecated inline path remains listed because pasted/manual row ingestion
  still uses it.

## Deferred

- Future PR: render file upload limits in `ContentOpsNewRun`.
- Future PR: remove inline compatibility after an operator compatibility
  window.
- Parked hardening: none.

## Verification

- Passed: focused control-surface catalog/limits pytest (`4 passed`).
- Passed: full control-surface API pytest:
  `python -m pytest tests/test_extracted_content_control_surface_api.py -q`
  (`86 passed`).
- Passed: UI build: `npm run build`.
- Passed: UI lint: `npm run lint`.
- Passed: `git diff --check`.
- Passed: local PR review via `scripts/local_pr_review.sh`.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan doc | ~80 |
| Backend catalog + tests | ~45 |
| Frontend types/mapper/fixture | ~70 |
| Contract docs | ~5 |
| **Total** | **~200** |

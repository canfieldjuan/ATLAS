# PR-Content-Ops-Import-Admission-Provider

## Why this slice exists

`FILECONCURRENCY-2` tracks the remaining uploaded-file import risk: the current
route gate is process-local. A multi-worker host can multiply the configured
admission window because each process owns its own in-memory gate.

The extracted package should not hard-code Redis, Postgres advisory locks, or a
durable queue implementation. The source fix is to move the shared-admission
decision to a host-owned provider seam while keeping the existing in-process
gate as the safe default.

## Scope (this PR)

Ownership lane: content-ops/backend-file-ingestion-validation

1. Add a host-injected ingestion import admission provider to the Content Ops
   control-surface router.
2. Use the provider before resolving the import pool for non-dry-run legacy and
   uploaded-file imports.
3. Preserve the current process-local gate when no provider is configured.
4. Add focused route tests for custom admission denial and release behavior.

### Files touched

- `extracted_content_pipeline/api/control_surfaces.py`
- `tests/test_extracted_content_control_surface_api.py`
- `plans/PR-Content-Ops-Import-Admission-Provider.md`

## Mechanism

`create_content_ops_control_surface_router(...)` gains an optional
`ingestion_import_admission_provider`. The provider returns a gate-like object
with async or sync `acquire()` and `release()` methods. Non-dry-run import
routes resolve that gate before the database pool is resolved.

When no provider is supplied, the existing `_ExecuteConcurrencyGate` remains the
default. When a provider denies admission, the route still returns the existing
machine-readable 429 reason:
`content_ops_ingestion_import_at_capacity`.

## Intentional

- No Redis/Postgres queue implementation in this slice. The extracted package
  exposes the production seam; hosts choose the distributed backend.
- Dry-run imports still bypass admission because they do not write.
- Existing response shape and default local behavior stay compatible.

## Deferred

- A concrete Atlas-host distributed admission provider remains future work.
- Durable background import jobs and queue visibility remain parked under
  `FILECONCURRENCY-2`.
- Parked hardening: `FILECONCURRENCY-2` remains open because this slice adds the
  seam but not a deployed shared gate.

## Verification

- Focused control-surface tests:
  - `python -m pytest tests/test_extracted_content_control_surface_api.py -q`
  - `92 passed`
- Router factory caller smoke tests:
  - `python -m pytest tests/test_smoke_content_ops_ingestion_file_route.py tests/test_smoke_content_ops_ingestion_file_route_inprocess_load.py tests/test_extracted_content_ops_live_execute_harness.py -q`
  - `15 passed`
- Local PR review:
  - `bash scripts/local_pr_review.sh --allow-dirty`

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan doc | 76 |
| Router admission provider seam | 86 |
| Route tests | 173 |
| Total | 335 |

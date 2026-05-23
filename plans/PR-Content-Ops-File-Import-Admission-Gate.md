# PR-Content-Ops-File-Import-Admission-Gate

## Why this slice exists

The uploaded-file route now has live write and scale validation, but the
concurrency probe surfaced the next production risk: enough simultaneous
non-dry-run imports can pressure the host database connection pool. The
production app mounts the Content Ops routes with the shared Atlas DB pool, so
the source-level mitigation belongs at the route layer before an import request
resolves or acquires the pool.

This slice adds a bounded admission gate for hosted ingestion imports. It is the
smallest end-to-end source fix for the surfaced issue: one active import can
hold the gate, a second write receives a deterministic 429, and dry-run
inspection/import paths remain side-effect-free.

## Scope (this PR)

Ownership lane: content-ops/backend-file-ingestion-validation

1. Add a configurable `ingestion_import_max_concurrency` limit to
   `ContentOpsControlSurfaceApiConfig`.
2. Apply that limit to non-dry-run ingestion imports before resolving the import
   pool provider.
3. Wire both the legacy JSON import route and uploaded-file import route through
   the same admission helper.
4. Add focused route tests for config validation and file-upload admission
   behavior.

### Files touched

- `extracted_content_pipeline/api/control_surfaces.py`
- `tests/test_extracted_content_control_surface_api.py`
- `plans/PR-Content-Ops-File-Import-Admission-Gate.md`

## Mechanism

`create_content_ops_control_surface_router` will instantiate a second
`_ExecuteConcurrencyGate` using the new
`ContentOpsControlSurfaceApiConfig.ingestion_import_max_concurrency` value.
Non-dry-run import routes will enter a shared helper before resolving
`opportunity_import_pool_provider` or scope. If the gate is full, the route
raises:

```python
HTTPException(
    status_code=429,
    detail={
        "reason": "content_ops_ingestion_import_at_capacity",
        "max_concurrency": ingestion_import_gate.max_concurrency,
    },
)
```

The helper releases the gate in `finally` after the import succeeds or fails.
`dry_run=True` bypasses the gate and still uses the existing object-backed dry
run path.

## Intentional

- This is an in-process route admission gate for the hosted app's shared pool,
  not a distributed queue. Cross-process queueing and durable background jobs
  are larger product-hardening work.
- The default limit is conservative and configurable instead of derived from
  Postgres `max_connections`; the extracted package should not inspect host DB
  settings.
- The smoke harness's 150-process pool-creation ceiling is not fixed here. That
  harness creates one asyncpg pool per process, which is a different shape from
  the hosted app's shared pool.

## Deferred

- Cross-process backpressure, durable import jobs, and shared queue visibility
  remain follow-up production hardening.
- A reusable in-process load runner that proves shared-pool admission against a
  real DB is deferred until after this route contract is in place.

## Verification

- Focused route tests:
  - `python -m pytest tests/test_extracted_content_control_surface_api.py -q`
  - `88 passed`
- Extracted package gauntlet:
  - `bash scripts/validate_extracted_content_pipeline.sh`
  - Passed.
  - `python extracted/_shared/scripts/forbid_atlas_reasoning_imports.py extracted_content_pipeline`
  - Passed.
  - `python scripts/audit_extracted_standalone.py --fail-on-debt`
  - Passed.
  - `bash scripts/check_ascii_python.sh`
  - Passed.
  - `bash scripts/run_extracted_pipeline_checks.sh`
  - `1871 passed, 1 skipped, 1 warning`
- Local PR review:
  - `bash scripts/local_pr_review.sh --allow-dirty`
  - Pending.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan doc | 105 |
| Router admission gate | 80 |
| Route tests | 96 |
| Total | 281 |

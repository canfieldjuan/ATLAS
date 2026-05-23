# PR-Content-Ops-Execute-Concurrency-Gate

## Why this slice exists

The FAQ stress probe showed correctness holds at large row counts, but hosted
execution still needs survivability controls. The API already caps inline
`source_material` uploads at 1,000 rows; the next source-level issue is
concurrent `/content-ops/execute` traffic.

This slice adds a thin request-level concurrency gate at the hosted execute
route so excess execute requests fail fast instead of piling onto database and
worker resources.

## Scope (this PR)

Ownership lane: content-ops/faq-generator-validation

1. Add a router-local execute concurrency limit to
   `ContentOpsControlSurfaceApiConfig`.
2. Apply that limit around the `/execute` route's runtime work.
3. Return a machine-readable `429` overload response when the gate is full.
4. Add focused API tests that prove overload rejection and release behavior.

### Files touched

- `plans/PR-Content-Ops-Execute-Concurrency-Gate.md`
- `extracted_content_pipeline/api/control_surfaces.py`
- `tests/test_extracted_content_control_surface_api.py`

## Mechanism

The control-surface router factory constructs one in-memory gate per router
instance. The execute endpoint validates the request body, attempts to
enter the gate, and returns `429` with `reason=content_ops_execute_at_capacity`
when the gate is full. A `finally` block releases the slot after success,
partial failure, or exception.

The default is intentionally small and configurable. This is a per-process
guard, not a distributed queue.

## Intentional

- No async job runner, retry/backoff, distributed queue, or database pool
  tuning in this slice.
- No new row-count cap; the route already limits inline `source_material` to
  1,000 rows.
- The gate covers all `/execute` outputs, not only FAQ, because the shared route
  and shared database pool are the constrained resources.

## Deferred

- Cross-process/global admission control remains deferred until there is a
  deployed multi-worker topology to target.
- Background job execution for large uploads remains deferred to a later
  hardening slice.

## Verification

- Passed: focused control-surface API tests:
  `tests/test_extracted_content_control_surface_api.py` (`86 passed` after
  rebasing over PR #861).
- Passed: extracted pipeline validation bundle:
  `scripts/run_extracted_pipeline_checks.sh` (`1860 passed, 1 skipped`;
  extracted reasoning checks also passed).
- Passed: post-rebase `bash scripts/local_pr_review.sh --allow-dirty`.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan doc | ~73 |
| Control-surface API | ~151 |
| API tests | ~66 |
| **Total** | **~291** |

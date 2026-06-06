# PR-Content-Ops-FAQ-Search-DB-Close-Result

## Why this slice exists

`PR-Content-Ops-FAQ-Search-DB-Cleanup-Result` fixed cleanup failures masking the
primary FAQ search DB smoke result, then parked the smaller adjacent lifecycle
edge: `pool.close()` can still raise from the `finally` block and replace an
already-built setup or search summary.

This production-hardening slice drains that parked edge so the smoke result
shows the primary outcome, cleanup status, and pool-close status separately.

## Scope (this PR)

Ownership lane: content-ops/faq-generator
Slice phase: Production hardening

1. Add pool-close status metadata to FAQ search DB smoke result payloads.
2. Convert `pool.close()` failures into `pool_close.error` metadata instead of
   letting them replace the original setup/search result.
3. Make pool-close failure fail the smoke overall while preserving the original
   `setup`, `cleanup`, `requests`, `latency`, and `isolation` fields.
4. Remove the drained pool-close hardening item from `HARDENING.md`.

### Files touched

| File | Purpose |
|---|---|
| `plans/PR-Content-Ops-FAQ-Search-DB-Close-Result.md` | Plan contract for the pool-close result hardening slice. |
| `HARDENING.md` | Remove the drained pool-close masking item. |
| `scripts/smoke_content_ops_faq_search_concurrency.py` | Preserve primary smoke results and report pool-close failures as metadata. |
| `tests/test_smoke_content_ops_faq_search_concurrency.py` | Regression tests for pool-close failure metadata on setup and search outcomes. |

## Mechanism

The smoke result already has lifecycle-style cleanup metadata. This slice uses
the same `{ok, attempted, error}` shape for a new `pool_close` result object.
`run_smoke(...)` catches `pool.close()` exceptions, records the type and message,
and applies the result to the already-built summary after cleanup metadata is
attached.

Overall `ok` becomes false when pool close fails, but the original setup/search
phase and request evidence remain visible.

## Intentional

- No cleanup SQL, search, seeding, routing, or result request behavior changes.
- `pool_close` is a separate top-level lifecycle object rather than overloading
  `cleanup`, because closing the pool is not data cleanup.
- This slice drains the last known result-masking edge in this smoke script.

## Deferred

- Parked hardening: none.

## Verification

- Command: pytest tests/test_smoke_content_ops_faq_search_concurrency.py -q
  - 26 passed.
- Command: python -m py_compile scripts/smoke_content_ops_faq_search_concurrency.py tests/test_smoke_content_ops_faq_search_concurrency.py
  - Passed.
- Command: python scripts/audit_plan_code_consistency.py plans/PR-Content-Ops-FAQ-Search-DB-Close-Result.md
  - Passed.
- Command: python scripts/audit_extracted_pipeline_ci_enrollment.py .
  - Passed: 121 matching tests enrolled.
- Command: git diff --check
  - Passed.
- Command: bash scripts/validate_extracted_content_pipeline.sh
  - Passed.
- Command: python extracted/_shared/scripts/forbid_atlas_reasoning_imports.py extracted_content_pipeline
  - Passed.
- Command: python scripts/audit_extracted_standalone.py --fail-on-debt
  - Passed.
- Command: bash scripts/check_ascii_python.sh
  - Passed.
- Command: bash scripts/run_extracted_pipeline_checks.sh
  - Passed: 2480 passed, 7 skipped, 1 warning.

## Estimated diff size

| File | LOC |
|---|---:|
| `plans/PR-Content-Ops-FAQ-Search-DB-Close-Result.md` | 87 |
| `HARDENING.md` | 14 |
| `scripts/smoke_content_ops_faq_search_concurrency.py` | 30 |
| `tests/test_smoke_content_ops_faq_search_concurrency.py` | 153 |
| **Total** | **284** |

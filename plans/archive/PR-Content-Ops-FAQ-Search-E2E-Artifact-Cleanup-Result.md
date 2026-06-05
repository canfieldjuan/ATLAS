# PR-Content-Ops-FAQ-Search-E2E-Artifact-Cleanup-Result

## Why this slice exists

The FAQ search DB smoke result-observability series now preserves setup, search,
cleanup, and pool-close outcomes. The adjacent seeded-route E2E smoke has one
similar lifecycle edge: when it uses its default temporary artifact directory,
`TemporaryDirectory.cleanup()` runs in `finally` and can replace an already-built
summary before `--output-result` is written.

This production-hardening slice preserves the E2E summary and reports temporary
artifact cleanup failures as metadata.

## Scope (this PR)

Ownership lane: content-ops/faq-generator
Slice phase: Production hardening

1. Add `artifact_cleanup` lifecycle metadata to seeded-route E2E smoke results.
2. Convert temporary artifact cleanup failures into `artifact_cleanup.error`
   metadata instead of letting them mask the primary seed/route/detail/cleanup
   summary.
3. Make artifact cleanup failure fail the smoke overall while preserving the
   original `seed`, `route`, `detail`, and `cleanup` fields.
4. Keep explicit `--artifact-dir` behavior unchanged; no cleanup is attempted
   for caller-owned artifact directories.
5. Include artifact cleanup status and error text in the default non-JSON
   summary output.

### Files touched

| File | Purpose |
|---|---|
| `plans/PR-Content-Ops-FAQ-Search-E2E-Artifact-Cleanup-Result.md` | Plan contract for the seeded-route artifact cleanup result slice. |
| `scripts/smoke_content_ops_faq_search_seeded_route_e2e.py` | Preserve E2E summaries and report temporary artifact cleanup failures as metadata. |
| `tests/test_smoke_content_ops_faq_search_seeded_route_e2e.py` | Regression tests for temporary artifact cleanup failure metadata. |

## Mechanism

`_run(...)` builds the same summary as today, but stores it before leaving the
`try` block. The `finally` catches temporary artifact cleanup failures into an
`artifact_cleanup` lifecycle object. After cleanup runs, `_run(...)` attaches the
lifecycle object and recomputes `ok` so cleanup failure is visible and fail-closed
without replacing the original E2E result.

Preflight summaries also include a not-attempted `artifact_cleanup` object so the
result envelope is stable. The default non-JSON summary prints artifact cleanup
status and the error type/message when artifact cleanup fails, so operators do
not need `--json` to see why the smoke exited non-zero.

## Intentional

- No seed, hosted route, detail route, DB cleanup, or artifact payload behavior
  changes.
- Caller-owned `--artifact-dir` paths are not cleaned up by this script and keep
  `artifact_cleanup.attempted = false`.
- This is scoped to temporary artifact cleanup only; command execution exceptions
  are not changed in this slice.

## Deferred

- Parked hardening: none.

## Verification

- Command: pytest tests/test_smoke_content_ops_faq_search_seeded_route_e2e.py -q
  - 49 passed.
- Command: python -m py_compile scripts/smoke_content_ops_faq_search_seeded_route_e2e.py tests/test_smoke_content_ops_faq_search_seeded_route_e2e.py
  - Passed.
- Command: python scripts/audit_plan_code_consistency.py plans/PR-Content-Ops-FAQ-Search-E2E-Artifact-Cleanup-Result.md
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
  - Passed: 2481 passed, 7 skipped, 1 warning.

## Estimated diff size

| File | LOC |
|---|---:|
| `plans/PR-Content-Ops-FAQ-Search-E2E-Artifact-Cleanup-Result.md` | 94 |
| `scripts/smoke_content_ops_faq_search_seeded_route_e2e.py` | 51 |
| `tests/test_smoke_content_ops_faq_search_seeded_route_e2e.py` | 66 |
| **Total** | **211** |

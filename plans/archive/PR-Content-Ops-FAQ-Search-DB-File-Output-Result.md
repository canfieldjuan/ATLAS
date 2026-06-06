# PR-Content-Ops-FAQ-Search-DB-File-Output-Result

## Why this slice exists

`PR-Content-Ops-FAQ-Search-DB-Setup-Failure-Result` made DB FAQ search smoke
pool, migration, and seed setup failures write structured `--output-result`
artifacts. It also parked one smaller adjacent gap: optional cleanup-manifest
and route-case file writes can still abort the smoke before the result artifact
is written.

This production-hardening slice drains that parked file-output gap while keeping
cleanup masking parked for a separate result-shape decision.

## Scope (this PR)

Ownership lane: content-ops/faq-generator
Slice phase: Production hardening

1. Convert cleanup-manifest write failures into a structured setup result with
   `setup.phase = "cleanup_manifest_output"`.
2. Convert route-case file write failures into a structured setup result with
   `setup.phase = "route_case_file_output"`.
3. Keep successful DB search execution, route-case payload shape, and cleanup
   policy unchanged.
4. Remove the drained file-output item from `HARDENING.md`; leave cleanup
   masking parked.

### Files touched

| File | Purpose |
|---|---|
| `plans/PR-Content-Ops-FAQ-Search-DB-File-Output-Result.md` | Plan contract for the file-output setup result slice. |
| `HARDENING.md` | Remove the drained optional file-output hardening item. |
| `scripts/smoke_content_ops_faq_search_concurrency.py` | Return structured result artifacts for optional file-output failures. |
| `tests/test_smoke_content_ops_faq_search_concurrency.py` | Regression tests for cleanup-manifest and route-case file output failures. |

## Mechanism

`run_smoke(...)` already has `_setup_failure_summary(...)` for setup phases.
This slice wraps `_write_cleanup_manifest(...)` and `_write_route_case_file(...)`
with narrow `try/except Exception` blocks. On failure, the smoke returns exit
code `1` with no request results and a phase-specific `setup.error` payload.

The existing `finally` block still closes the pool and runs cleanup after
migrations succeed unless `--keep-data` is set.

## Intentional

- No new result shape is introduced; file-output failures use the same setup
  summary used by pool, migration, and seed failures.
- Cleanup failures are not fixed here. Modeling cleanup as metadata instead of
  masking the original result is a separate, larger hardening slice.
- Route-case and cleanup-manifest payload schemas are unchanged.

## Deferred

- Parked hardening: cleanup failures in
  `scripts/smoke_content_ops_faq_search_concurrency.py` can still mask the
  original setup/search result.

## Verification

- Command: pytest tests/test_smoke_content_ops_faq_search_concurrency.py -q
  - 22 passed.
- Command: python -m py_compile scripts/smoke_content_ops_faq_search_concurrency.py tests/test_smoke_content_ops_faq_search_concurrency.py
  - Passed.
- Command: python scripts/audit_plan_code_consistency.py plans/PR-Content-Ops-FAQ-Search-DB-File-Output-Result.md
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
  - Passed: 2469 passed, 7 skipped, 1 warning.

## Estimated diff size

| File | LOC |
|---|---:|
| `plans/PR-Content-Ops-FAQ-Search-DB-File-Output-Result.md` | 92 |
| `HARDENING.md` | 13 |
| `scripts/smoke_content_ops_faq_search_concurrency.py` | 32 |
| `tests/test_smoke_content_ops_faq_search_concurrency.py` | 134 |
| **Total** | **271** |

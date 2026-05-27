# PR-Content-Ops-FAQ-Search-DB-Cleanup-Result

## Why this slice exists

`PR-Content-Ops-FAQ-Search-DB-Setup-Failure-Result` parked a remaining FAQ
search DB smoke survivability gap: cleanup failures in `run_smoke(...)` can mask
an earlier setup or search result because cleanup runs in `finally` after the
smoke has already prepared a return value.

This production-hardening slice drains that gap at the result-envelope source,
so go-live operators can see both the primary setup/search outcome and whether
cleanup succeeded.

## Scope (this PR)

Ownership lane: content-ops/faq-generator
Slice phase: Production hardening

1. Add cleanup status metadata to FAQ search DB smoke result payloads.
2. Convert `_cleanup(...)` failures into `cleanup.error` metadata instead of
   letting them replace the original setup/search result.
3. Make cleanup failure fail the smoke overall while preserving the original
   `setup`, `requests`, `latency`, and `isolation` fields.
4. Remove the drained cleanup-masking item from `HARDENING.md` and park the
   smaller pool-close lifecycle edge surfaced during review.

### Files touched

| File | Purpose |
|---|---|
| `plans/PR-Content-Ops-FAQ-Search-DB-Cleanup-Result.md` | Plan contract for the cleanup-result hardening slice. |
| `HARDENING.md` | Remove the drained cleanup masking item and park the pool-close lifecycle edge. |
| `scripts/smoke_content_ops_faq_search_concurrency.py` | Preserve primary smoke results and report cleanup failures as metadata. |
| `tests/test_smoke_content_ops_faq_search_concurrency.py` | Regression tests for cleanup failure metadata on setup and search outcomes. |

## Mechanism

`_summary_payload(...)` gains a default not-attempted `cleanup` result object,
so pool/preflight failures remain simple. When cleanup is attempted,
`run_smoke(...)` catches `_cleanup(...)` exceptions, records their error type
and message, and returns the already-built summary with cleanup metadata
attached.

Overall `ok` becomes false when cleanup fails, but the original setup/search
phase and request evidence remain visible.

## Intentional

- This slice handles `_cleanup(...)` failures only; pool close failures are a
  separate lifecycle edge and are not introduced by this change.
- No search, seeding, routing, or cleanup SQL behavior changes.
- Cleanup failure is treated as an overall smoke failure because leaked smoke
  rows are operationally actionable even when retrieval itself passed.

## Deferred

- Parked hardening: pool close failures in
  `scripts/smoke_content_ops_faq_search_concurrency.py` can still mask the
  original setup/search result.

## Verification

- Command: pytest tests/test_smoke_content_ops_faq_search_concurrency.py -q
  - 24 passed.
- Command: python -m py_compile scripts/smoke_content_ops_faq_search_concurrency.py tests/test_smoke_content_ops_faq_search_concurrency.py
  - Passed.
- Command: python scripts/audit_plan_code_consistency.py plans/PR-Content-Ops-FAQ-Search-DB-Cleanup-Result.md
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
  - Passed: 2478 passed, 7 skipped, 1 warning.

## Estimated diff size

| File | LOC |
|---|---:|
| `plans/PR-Content-Ops-FAQ-Search-DB-Cleanup-Result.md` | 92 |
| `HARDENING.md` | 14 |
| `scripts/smoke_content_ops_faq_search_concurrency.py` | 140 |
| `tests/test_smoke_content_ops_faq_search_concurrency.py` | 149 |
| **Total** | **395** |

# PR-Content-Ops-FAQ-Search-DB-Setup-Failure-Result

## Why this slice exists

PR-Content-Ops-FAQ-Search-DB-Preflight-Result made malformed DB FAQ search
smoke invocations write `--output-result`, but setup failures after pool
creation can still abort before the JSON artifact is written. Migration and
seed failures are the next highest-value setup edges because they happen before
retrieval starts and are the failures an operator most needs to distinguish
from query latency or tenant-isolation failures.

This production-hardening slice makes migration and seed setup failures visible
without changing the successful DB search path, hosted route path, or cleanup
policy.

## Scope (this PR)

Ownership lane: content-ops/faq-generator
Slice phase: Production hardening

1. Convert migration failures into a structured DB FAQ search result with
   `setup.phase = "migrations"`.
2. Convert seed failures into a structured DB FAQ search result with
   `setup.phase = "seed"`.
3. Keep the search execution, latency gates, route-case generation, and cleanup
   policy unchanged.
4. Add focused negative tests for both setup-failure branches.

### Files touched

| File | Purpose |
|---|---|
| `plans/PR-Content-Ops-FAQ-Search-DB-Setup-Failure-Result.md` | Plan contract for the setup-failure result slice. |
| `HARDENING.md` | Park adjacent optional file-output and cleanup-result visibility gaps. |
| `scripts/smoke_content_ops_faq_search_concurrency.py` | Return structured result artifacts for migration and seed failures. |
| `tests/test_smoke_content_ops_faq_search_concurrency.py` | Regression tests for migration and seed failure result artifacts. |

## Mechanism

Add a small `_setup_failure_summary(...)` helper that uses the existing
`_summary_payload(...)` result shape with `ok = false`, no request results, and
a phase-specific `setup.error` block. `run_smoke(...)` catches exceptions around
only `_apply_migrations(...)` and `_seed(...)`, returns exit code `1`, and then
lets the existing `finally` cleanup/close path run.

The successful path still runs migrations, writes optional case/cleanup files,
seeds documents, runs concurrent searches, applies latency budgets, and returns
the existing success summary.

## Intentional

- No broad catch around the retrieval phase; worker failures are already
  represented in `isolation`.
- No cleanup behavior change. Cleanup still runs in `finally` unless
  `--keep-data` is set.
- Optional manifest and route-case file write failures are parked instead of
  included so this PR stays focused on DB setup phases.

## Deferred

- Parked hardening: optional file-output failures in
  `scripts/smoke_content_ops_faq_search_concurrency.py` can still abort before
  result artifact writing.
- Parked hardening: cleanup failures in
  `scripts/smoke_content_ops_faq_search_concurrency.py` can still mask the
  original setup/search result.

## Verification

- `pytest tests/test_smoke_content_ops_faq_search_concurrency.py -q` - 20
  passed.
- Py compile for `scripts/smoke_content_ops_faq_search_concurrency.py` and
  `tests/test_smoke_content_ops_faq_search_concurrency.py` - passed.
- Plan/code consistency audit for
  `plans/PR-Content-Ops-FAQ-Search-DB-Setup-Failure-Result.md` - passed.
- Extracted pipeline CI enrollment audit - 121 matching tests enrolled.
- `git diff --check` - passed.
- Extracted content pipeline validation script - passed.
- `extracted/_shared/scripts/forbid_atlas_reasoning_imports.py` for
  `extracted_content_pipeline` - passed.
- Extracted standalone audit with `--fail-on-debt` - passed.
- Python ASCII check for extracted packages - passed.
- Extracted pipeline CI mirror - 2467 passed, 7 skipped, 1 warning.

## Estimated diff size

| File | LOC |
|---|---:|
| `plans/PR-Content-Ops-FAQ-Search-DB-Setup-Failure-Result.md` | 93 |
| `HARDENING.md` | 27 |
| `scripts/smoke_content_ops_faq_search_concurrency.py` | 51 |
| `tests/test_smoke_content_ops_faq_search_concurrency.py` | 126 |
| **Total** | **297** |

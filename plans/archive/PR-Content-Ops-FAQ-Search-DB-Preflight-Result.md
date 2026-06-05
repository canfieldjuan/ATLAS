# PR-Content-Ops-FAQ-Search-DB-Preflight-Result

## Why this slice exists

The seeded hosted FAQ search e2e runner now exposes not-run reasons for seed,
route, and detail phases, but the lower-level DB FAQ search concurrency smoke
still exits during argument preflight before writing `--output-result`. That
means a malformed go-live invocation can leave operators with no JSON artifact
to inspect, which is inconsistent with the rest of the FAQ search smoke lane.

This production-hardening slice makes preflight failures observable at the DB
smoke boundary without changing the seeded search, cleanup, hosted route, or
latency behavior.

## Scope (this PR)

Ownership lane: content-ops/faq-generator
Slice phase: Production hardening

1. Write a deterministic preflight result payload when DB FAQ search smoke
   argument validation fails.
2. Keep the existing validation rules and DB/search execution path unchanged.
3. Add a focused negative test proving `--output-result` is written on a
   preflight failure.

### Files touched

| File | Purpose |
|---|---|
| `plans/PR-Content-Ops-FAQ-Search-DB-Preflight-Result.md` | Plan contract for the preflight result hardening slice. |
| `scripts/smoke_content_ops_faq_search_concurrency.py` | Emit a JSON result artifact for validation failures. |
| `tests/test_smoke_content_ops_faq_search_concurrency.py` | Regression test for the preflight artifact path. |

## Mechanism

`main()` already owns argument parsing, output-result writing, summary printing,
and exit-code translation. The change catches `SystemExit` raised by
`run_smoke()` validation, builds a summary through the same `_summary_payload`
shape with `setup.phase = "preflight"`, writes `--output-result`, prints the
summary, and returns exit code `2`.

The normal path still calls `asyncio.run(run_smoke(args))`; pool creation
failures, migration/seed/search behavior, cleanup, and latency gates are not
changed.

## Intentional

- The validation function keeps raising `SystemExit`; this avoids refactoring
  direct validation tests and keeps the slice focused on result visibility.
- Preflight exits return `2`, matching the hosted route concurrency smoke's
  preflight behavior.
- No new hardening is added for migration/seed exceptions in this slice; the
  current target is preflight artifact visibility only.

## Deferred

- Future production-hardening slice: decide whether DB migration/seed failures
  should also be converted into structured setup phases before cleanup.
- Parked hardening: none. `HARDENING.md` has no active FAQ search items touching
  this script.

## Verification

- `pytest tests/test_smoke_content_ops_faq_search_concurrency.py -q` - 18
  passed.
- Py compile for `scripts/smoke_content_ops_faq_search_concurrency.py` and
  `tests/test_smoke_content_ops_faq_search_concurrency.py` - passed.
- Plan/code consistency audit for
  `plans/PR-Content-Ops-FAQ-Search-DB-Preflight-Result.md` - passed.
- Extracted pipeline CI enrollment audit - 121 matching tests enrolled.
- `git diff --check` - passed.
- Extracted content pipeline validation script - passed.
- `extracted/_shared/scripts/forbid_atlas_reasoning_imports.py` for
  `extracted_content_pipeline` - passed.
- Extracted standalone audit with `--fail-on-debt` - passed.
- Python ASCII check for extracted packages - passed.
- Extracted pipeline CI mirror - 2465 passed, 7 skipped, 1 warning.

## Estimated diff size

| File | LOC |
|---|---:|
| `plans/PR-Content-Ops-FAQ-Search-DB-Preflight-Result.md` | 86 |
| `scripts/smoke_content_ops_faq_search_concurrency.py` | 42 |
| `tests/test_smoke_content_ops_faq_search_concurrency.py` | 68 |
| **Total** | **196** |

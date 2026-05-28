# PR-Content-Ops-FAQ-Search-E2E-Case-Budgets

## Why this slice exists

Recent FAQ route-concurrency slices added per-case error and latency budgets to
the hosted route smoke, including fail-closed handling when a budgeted case has
no samples. The seeded hosted e2e runner composes that route smoke, but it still
only exposes the aggregate route budgets. That wrapper drift lets the real
seeded flow miss the stricter case-level gates operators now use to prove mixed
corpora remain healthy under load.

## Scope (this PR)

Ownership lane: content-ops/faq-search
Slice phase: Production hardening

1. Add seeded e2e CLI flags for the route smoke's per-case error and latency
   budgets.
2. Validate those budgets in the seeded e2e preflight with the same bounds as
   the child route smoke.
3. Forward those budgets into the route concurrency child command.
4. Add focused tests for validation, parser wiring, and command handoff.

### Files touched

| File | Purpose |
|---|---|
| `plans/PR-Content-Ops-FAQ-Search-E2E-Case-Budgets.md` | Plan contract for this wrapper-drift fix. |
| `scripts/smoke_content_ops_faq_search_seeded_route_e2e.py` | Expose, validate, and forward per-case route budgets. |
| `tests/test_smoke_content_ops_faq_search_seeded_route_e2e.py` | Prove the seeded e2e wrapper accepts and forwards the budgets, and rejects invalid values. |

## Mechanism

The seeded e2e parser gains:

```text
--max-case-error-rate
--max-case-p95-ms
--max-case-single-request-ms
```

`_validate_args(...)` checks `--max-case-error-rate` is between 0 and 1 and
checks the two per-case latency values are positive when provided. The route
child command appends the flags only when operators pass them, preserving the
existing default behavior.

## Intentional

- No hosted route, database seed, cleanup, or detail-check behavior changes.
- No default SLO values are introduced; budget selection remains operator-owned.
- `--max-detail-ms` is not added to the seeded e2e wrapper in this slice because
  route detail concurrency is not enabled by the seeded e2e route phase. The
  seeded detail phase already uses the single-detail contract checker.

## Deferred

- Parked hardening: none. `HARDENING.md` was scanned and has no active FAQ
  search entries touching this wrapper.
- Detail-specific budget threading for the seeded detail phase remains deferred
  until operators have a concrete hosted detail latency target for that command.

## Verification

- `python -m pytest tests/test_smoke_content_ops_faq_search_seeded_route_e2e.py -q` - 56 passed.
- `python -m py_compile scripts/smoke_content_ops_faq_search_seeded_route_e2e.py tests/test_smoke_content_ops_faq_search_seeded_route_e2e.py` - passed.
- `python scripts/audit_plan_doc.py plans/PR-Content-Ops-FAQ-Search-E2E-Case-Budgets.md` - passed.
- `python scripts/audit_plan_code_consistency.py plans/PR-Content-Ops-FAQ-Search-E2E-Case-Budgets.md` - passed.
- `python scripts/audit_extracted_pipeline_ci_enrollment.py .` - 122 matching tests enrolled.
- `git diff --check` - passed.
- `bash scripts/validate_extracted_content_pipeline.sh` - passed.
- `python extracted/_shared/scripts/forbid_atlas_reasoning_imports.py extracted_content_pipeline` - passed.
- `python scripts/audit_extracted_standalone.py --fail-on-debt` - passed.
- `bash scripts/check_ascii_python.sh` - passed.
- `bash scripts/run_extracted_pipeline_checks.sh` - 2570 passed, 7 skipped, 1 warning.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan doc | 83 |
| Seeded e2e runner | 28 |
| Tests | 60 |
| **Total** | **171** |

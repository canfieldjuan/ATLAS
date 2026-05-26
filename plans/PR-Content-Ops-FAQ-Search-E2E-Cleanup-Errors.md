# PR-Content-Ops-FAQ-Search-E2E-Cleanup-Errors

## Why this slice exists

The FAQ search seeded e2e runner now has clearer detail-phase visibility, but
its cleanup phase still reports failures with a scalar `error` field. Recent FAQ
operator artifacts use `errors: list[str]` so callers can consume one failure
surface across modes and phases. This slice aligns the seeded e2e cleanup
payload with that convention.

## Scope (this PR)

Ownership lane: content-ops/faq-search
Slice phase: Production hardening

1. Change seeded e2e cleanup summaries from scalar `error` to
   `errors: list[str]`.
2. Normalize all cleanup branches: no IDs, import failure, database exception,
   malformed delete status, rowcount mismatch, manifest parse failure, and
   success.
3. Update focused tests to assert the new cleanup envelope.

### Files touched

- `plans/PR-Content-Ops-FAQ-Search-E2E-Cleanup-Errors.md`
- `scripts/smoke_content_ops_faq_search_seeded_route_e2e.py`
- `tests/test_smoke_content_ops_faq_search_seeded_route_e2e.py`

## Mechanism

A small `_cleanup_result(...)` helper builds the cleanup phase payload with
`ok = not errors`, row counts, delete status, and `errors`. Existing cleanup
call sites keep their control flow but pass failure messages as list entries.
The route/detail/seed phases are unchanged.

## Intentional

- No database query change; the cleanup DELETE and cascade behavior stay the
  same.
- No hosted-route behavior change; this is result-artifact hardening.
- No broad e2e result-envelope refactor beyond the cleanup phase.

## Deferred

Parked hardening: none. `HARDENING.md` has no active FAQ search entries touching
this runner.

Route phase result normalization is left for a later slice if it becomes a
consumer pain point; this slice only closes the scalar cleanup error drift.

## Verification

- `pytest tests/test_smoke_content_ops_faq_search_seeded_route_e2e.py -q` (47
  passed)
- `python -m py_compile` with
  `scripts/smoke_content_ops_faq_search_seeded_route_e2e.py` and
  `tests/test_smoke_content_ops_faq_search_seeded_route_e2e.py`
- `python` `scripts/audit_plan_code_consistency.py` with
  `plans/PR-Content-Ops-FAQ-Search-E2E-Cleanup-Errors.md`
- `python scripts/audit_extracted_pipeline_ci_enrollment.py .` (121 matching
  tests enrolled)
- `git diff --check`
- `bash` `scripts/validate_extracted_content_pipeline.sh`
- `python extracted/_shared/scripts/forbid_atlas_reasoning_imports.py
  extracted_content_pipeline`
- `python scripts/audit_extracted_standalone.py --fail-on-debt`
- `bash` `scripts/check_ascii_python.sh`
- `bash` `scripts/run_extracted_pipeline_checks.sh` (2456 passed, 6 skipped)

## Estimated diff size

| Area | Estimated LOC |
| --- | ---: |
| Plan doc | 77 |
| Runner | 101 |
| Tests | 18 |
| **Total** | **196** |

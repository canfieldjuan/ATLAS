# PR-Content-Ops-FAQ-Search-Detail-Not-Run-Reason

## Why this slice exists

The seeded hosted FAQ search e2e runner now checks search and detail hydration,
but its result JSON can make an unattempted detail check look like a detail
contract failure. When seed or route fails first, `detail` stays `ok=false`
without saying that the phase was never run. Operators need that distinction
while debugging hosted demo readiness.

## Scope (this PR)

Ownership lane: content-ops/faq-search
Slice phase: Production hardening

1. Add an explicit `not_run_reason` to seeded e2e detail summaries when the
   detail phase is skipped or blocked by an earlier phase.
2. Preserve existing pass/fail behavior: skipped-by-flag detail remains
   non-blocking, while seed/route/preflight-blocked detail remains failed.
3. Add focused tests for skip-by-flag and route-failure detail visibility.

### Files touched

- `plans/PR-Content-Ops-FAQ-Search-Detail-Not-Run-Reason.md`
- `scripts/smoke_content_ops_faq_search_seeded_route_e2e.py`
- `tests/test_smoke_content_ops_faq_search_seeded_route_e2e.py`

## Mechanism

A small helper builds the canonical detail-not-run payload:
`ok`, `returncode`, `skipped`, and `not_run_reason`. Preflight, explicit
`--skip-detail-check`, seed failure, and route failure use that helper. The
actual detail contract checker path is unchanged.

## Intentional

- No hosted route behavior changes; this only improves the smoke result
  artifact.
- No cleanup behavior changes; cleanup still runs after seed/route failures
  whenever a manifest exists.
- No broad result-envelope refactor in this slice.

## Deferred

Parked hardening: none. `HARDENING.md` has no active FAQ search items touching
this runner.

Route not-run visibility can be handled in a separate slice if operators need
the same explicit phase reason there; this slice is limited to the detail
ambiguity already called out in the FAQ search plans.

## Verification

- `pytest tests/test_smoke_content_ops_faq_search_seeded_route_e2e.py -q` (47
  passed)
- `python -m py_compile` with
  `scripts/smoke_content_ops_faq_search_seeded_route_e2e.py` and
  `tests/test_smoke_content_ops_faq_search_seeded_route_e2e.py`
- `python` `scripts/audit_plan_code_consistency.py` with
  `plans/PR-Content-Ops-FAQ-Search-Detail-Not-Run-Reason.md`
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
| Plan doc | 78 |
| Runner | 31 |
| Tests | 23 |
| **Total** | **132** |

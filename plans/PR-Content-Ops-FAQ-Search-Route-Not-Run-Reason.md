# PR-Content-Ops-FAQ-Search-Route-Not-Run-Reason

## Why this slice exists

The seeded hosted FAQ search e2e runner now makes detail not-run states explicit,
but the route phase still looks like a route failure when it was never attempted
because preflight or seed failed first. That can send an operator toward hosted
route debugging when the actual blocker is earlier in the e2e flow.

## Scope (this PR)

Ownership lane: content-ops/faq-search
Slice phase: Production hardening

1. Add an explicit route not-run payload with `skipped: true` and
   `not_run_reason`.
2. Use that payload for preflight and seed-failure route states.
3. Add focused tests that distinguish route not-run from a real route command
   failure.

### Files touched

- `plans/PR-Content-Ops-FAQ-Search-Route-Not-Run-Reason.md`
- `scripts/smoke_content_ops_faq_search_seeded_route_e2e.py`
- `tests/test_smoke_content_ops_faq_search_seeded_route_e2e.py`

## Mechanism

A `_route_not_run(...)` helper builds the route phase payload:
`ok`, `returncode`, `stdout_tail`, `stderr_tail`, `skipped`, and
`not_run_reason`. Preflight and seed-failure paths use that helper. When the
route command actually runs, `_run_command(...)` remains unchanged, so real
hosted-route failures still carry the command return code and output tails.

## Intentional

- No hosted route behavior changes; this only improves the e2e result artifact.
- No cleanup behavior changes.
- No broad seed/detail/result refactor in this slice.

## Deferred

Parked hardening: none. `HARDENING.md` has no active FAQ search entries touching
this runner.

Seed phase result normalization is left for a later slice if it becomes a
consumer pain point; this slice only closes route-phase ambiguity.

## Verification

- `pytest tests/test_smoke_content_ops_faq_search_seeded_route_e2e.py -q` (48
  passed)
- `python -m py_compile` with
  `scripts/smoke_content_ops_faq_search_seeded_route_e2e.py` and
  `tests/test_smoke_content_ops_faq_search_seeded_route_e2e.py`
- `python` `scripts/audit_plan_code_consistency.py` with
  `plans/PR-Content-Ops-FAQ-Search-Route-Not-Run-Reason.md`
- `python scripts/audit_extracted_pipeline_ci_enrollment.py .` (121 matching
  tests enrolled)
- `git diff --check`
- `bash` `scripts/validate_extracted_content_pipeline.sh`
- `python extracted/_shared/scripts/forbid_atlas_reasoning_imports.py
  extracted_content_pipeline`
- `python scripts/audit_extracted_standalone.py --fail-on-debt`
- `bash` `scripts/check_ascii_python.sh`
- `bash` `scripts/run_extracted_pipeline_checks.sh` (2460 passed, 6 skipped)

## Estimated diff size

| Area | Estimated LOC |
| --- | ---: |
| Plan doc | 75 |
| Runner | 19 |
| Tests | 71 |
| **Total** | **165** |

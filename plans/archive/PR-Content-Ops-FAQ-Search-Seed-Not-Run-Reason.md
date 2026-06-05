# PR-Content-Ops-FAQ-Search-Seed-Not-Run-Reason

## Why this slice exists

The seeded hosted FAQ search e2e runner now marks route and detail not-run
states explicitly. The seed phase has one remaining ambiguous not-run state:
preflight failure. Its current payload is only `{ok: false, returncode: null}`,
which is enough for the gate but less clear for operator debugging.

## Scope (this PR)

Ownership lane: content-ops/faq-search
Slice phase: Production hardening

1. Add an explicit seed not-run payload with `skipped: true` and
   `not_run_reason`.
2. Use that payload for preflight failures.
3. Add a focused assertion that preflight blocks seed rather than producing a
   seed command failure.

### Files touched

- `plans/PR-Content-Ops-FAQ-Search-Seed-Not-Run-Reason.md`
- `scripts/smoke_content_ops_faq_search_seeded_route_e2e.py`
- `tests/test_smoke_content_ops_faq_search_seeded_route_e2e.py`

## Mechanism

A `_seed_not_run(...)` helper builds the preflight seed-phase payload:
`ok`, `returncode`, `stdout_tail`, `stderr_tail`, `skipped`, and
`not_run_reason`. The actual seed command path still uses `_run_command(...)`,
so real seed failures keep the command return code and output tails.

## Intentional

- No database, hosted-route, cleanup, route, or detail behavior changes.
- No broad e2e result-envelope refactor beyond the seed preflight state.
- The seed command itself is not wrapped or changed.

## Deferred

Parked hardening: none. `HARDENING.md` has no active FAQ search entries touching
this runner.

No further route/detail cleanup is included; this slice only closes the
remaining seed not-run ambiguity.

## Verification

- `pytest tests/test_smoke_content_ops_faq_search_seeded_route_e2e.py -q` (48
  passed)
- `python -m py_compile` with
  `scripts/smoke_content_ops_faq_search_seeded_route_e2e.py` and
  `tests/test_smoke_content_ops_faq_search_seeded_route_e2e.py`
- `python` `scripts/audit_plan_code_consistency.py` with
  `plans/PR-Content-Ops-FAQ-Search-Seed-Not-Run-Reason.md`
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
| Plan doc | 74 |
| Runner | 13 |
| Tests | 8 |
| **Total** | **95** |

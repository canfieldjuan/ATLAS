# PR-Extracted-Pipeline-CI-Scanner-Fixtures

## Why this slice exists

PR-Extracted-Pipeline-CI-Enrollment-Scanner closed the live extracted pipeline
test-enrollment gap, but the scanner parser still lives inside the repo-state
test. That proves today's files are enrolled, but it does not give the parser
its own fixture coverage for the failure shapes it is meant to catch.

AGENTS.md section 3h says new auditors should ship with fixture tests. This
slice turns the enrollment scanner into a small audit module with fixture tests
so future edits cannot silently regress to false-green parsing.

This is above the normal diff-size target because the extraction, fixture
coverage, live contract update, and workflow enrollment have to move together:
shipping the module without enrolled fixture tests would recreate the same
non-gating pattern this slice is closing.

## Scope (this PR)

Ownership lane: workflow/extracted-pipeline-ci

Slice phase: Workflow/process

1. Extract the CI enrollment scanner helpers into
   `scripts/audit_extracted_pipeline_ci_enrollment.py`.
2. Keep `tests/test_extracted_pipeline_route_ci_contract.py` as the live
   repo-state contract that checks the real runner and workflow.
3. Add fixture tests for the extracted scanner's happy path and failure paths.
4. Enroll the new audit fixture test in the extracted pipeline runner and
   workflow path filters.

### Files touched

- `plans/PR-Extracted-Pipeline-CI-Scanner-Fixtures.md`
- `scripts/audit_extracted_pipeline_ci_enrollment.py`
- `tests/test_audit_extracted_pipeline_ci_enrollment.py`
- `tests/test_extracted_pipeline_route_ci_contract.py`
- `scripts/run_extracted_pipeline_checks.sh`
- `.github/workflows/extracted_pipeline_checks.yml`

## Mechanism

The new audit module keeps the same behavior introduced by the scanner:

- collect candidate tests from curated extracted/content-ops patterns;
- parse literal test paths from `scripts/run_extracted_pipeline_checks.sh`;
- parse quoted workflow path filters for both `pull_request` and `push`;
- report missing runner enrollment, missing workflow filter coverage, and a
  zero-candidate scan as failures.

The live contract test calls the audit module against the repository root. The
new fixture test creates temporary fake repos so parser behavior is tested
without mutating the real checkout.

## Intentional

- No product code changes.
- No broad workflow refactor.
- No switch to blanket-glob execution in the extracted runner; explicit
  enrollment stays the operating model.
- The audit fixture test is enrolled in the extracted runner because otherwise
  this slice would add an auditor whose own tests do not gate the workflow it
  protects.

## Deferred

- Future PR: consider wiring the audit module's CLI into local review if this
  parser starts protecting more than the extracted pipeline lane.
- Parked hardening: none.

## Verification

- Audit CLI `scripts/audit_extracted_pipeline_ci_enrollment.py`
  - passed, 112 matching tests enrolled.
- `python -m pytest tests/test_audit_extracted_pipeline_ci_enrollment.py -q`
  - passed, 8 tests.
- `python -m pytest tests/test_audit_extracted_pipeline_ci_enrollment.py tests/test_extracted_pipeline_route_ci_contract.py -q`
  - passed, 11 tests.
- Extracted pipeline runner `scripts/run_extracted_pipeline_checks.sh` passed.
  - 2148 pytest cases passed, 3 skipped, 1 existing environment warning from
    `torch`/`pynvml`.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan | ~95 |
| Audit module | ~165 |
| Fixture tests | ~160 |
| Repo-state contract update | ~100 |
| Workflow and runner enrollment | ~5 |
| **Total** | **~525** |

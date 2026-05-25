# PR-Extracted-CI-Enrollment-Checker-Evaluator-Patterns

## Why this slice exists
PR #955's review caught that the extracted CI enrollment scanner does not enforce
two test naming families that are now part of the content-ops support surface:
tests/test_check_content_ops*.py checker tests and tests/test_evaluate_*.py
evaluator tests. The checker test is manually enrolled today, but the scanner
would not detect a future removal. Adding the evaluator pattern also reveals the
existing support-ticket evaluator test needs explicit runner/workflow enrollment.

## Scope (this PR)
Ownership lane: content-ops/faq-search

Slice phase: Functional validation.

1. Add checker and evaluator test patterns to the extracted CI enrollment
   scanner.
2. Add scanner fixture coverage proving those patterns are candidates.
3. Enroll the existing support-ticket generated-content evaluator test in the
   extracted runner and workflow filters so the scanner passes.
4. Add workflow trigger globs for the same checker/evaluator naming families so
   a future new file runs the audit before merge.

### Files touched

- `plans/PR-Extracted-CI-Enrollment-Checker-Evaluator-Patterns.md`
- `scripts/audit_extracted_pipeline_ci_enrollment.py`
- `tests/test_audit_extracted_pipeline_ci_enrollment.py`
- `scripts/run_extracted_pipeline_checks.sh`
- `.github/workflows/extracted_pipeline_checks.yml`

## Mechanism
The scanner's `DEFAULT_ENROLLED_TEST_PATTERNS` gains the checker and evaluator
test globs. The fixture
test creates one file in each family and verifies `audit_ci_enrollment(...)`
treats them as candidates only when runner and workflow path filters include
them. The production runner and workflow gain the already-existing
`tests/test_evaluate_support_ticket_generated_content.py` path, and the workflow
path filters gain the checker/evaluator globs so future files trigger the audit.

## Intentional
- This is scanner/enrollment only; no evaluator or FAQ search checker behavior
  changes.
- The evaluator path is explicit in the runner, matching the current extracted
  gauntlet style instead of introducing glob execution.
- The workflow keeps the existing explicit checker/evaluator file paths while
  adding family globs that trigger the audit for future files.

## Deferred
- Generic naming policy documentation for future checker/evaluator tests is left
  to a workflow/process slice if this pattern recurs outside extracted CI.

## Verification
- pytest tests/test_audit_extracted_pipeline_ci_enrollment.py -q passed with 9 tests.
- pytest tests/test_evaluate_support_ticket_generated_content.py -q passed with 9 tests.
- python scripts/audit_extracted_pipeline_ci_enrollment.py passed and reported 114 matching tests enrolled.
- python -m compileall scripts/audit_extracted_pipeline_ci_enrollment.py tests/test_audit_extracted_pipeline_ci_enrollment.py passed.
- After review feedback, reran the same focused commands with workflow trigger
  globs added; all passed.
- Pending local run: bash scripts/local_pr_review.sh

## Estimated diff size
| Area | Estimated LOC |
|---|---:|
| Plan doc | 60 |
| Scanner + tests | 45 |
| Runner/workflow enrollment | 8 |
| **Total** | **113** |

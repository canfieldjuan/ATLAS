# PR-Reviewer-R14-Linter-CI-Enrollment

## Why this slice exists

#1472 added the R14 review-body linter and a workflow test proving the
pre-push audit passes PR-body context in CI. The reviewer caught one remaining
CI-enrollment gap before we moved on: the linter's own
`tests/test_check_review_body_r14.py` suite is not in the
`pre_push_audit.yml` tooling pytest list, so `check_review_body_r14.py` could
regress while CI stays green.

This slice closes that gap directly and completes the #1472 teardown by moving
the merged plan out of root `plans/` into `plans/archive/`.

## Scope (this PR)

Ownership lane: workflow/process
Slice phase: Workflow/process

1. Enroll `tests/test_check_review_body_r14.py` in the
   `.github/workflows/pre_push_audit.yml` PR-review tooling pytest list.
2. Archive the merged #1472 plan doc and refresh `plans/INDEX.md`.
3. Keep the linter implementation unchanged; this is CI coverage wiring only.

### Review Contract

- Acceptance criteria:
  - [ ] The pre-push audit workflow's PR-review tooling unit-test command runs
        `tests/test_check_review_body_r14.py`.
  - [ ] The workflow still runs the existing tooling tests, including
        `tests/test_pre_push_audit_workflow.py`.
  - [ ] The R14 linter's focused suite passes locally.
  - [ ] The merged #1472 plan is archived and no longer remains as an in-flight
        root plan file.
- Affected surfaces: pre-push audit workflow test list and plans archive only.
- Risk areas: recurring CI-enrollment misses / accidental removal of existing
  workflow tooling tests / plan archive index drift.
- Reviewer rules triggered: R2.

### Files touched

- `.github/workflows/pre_push_audit.yml`
- `plans/INDEX.md`
- `plans/PR-Reviewer-R14-Linter-CI-Enrollment.md`
- `plans/archive/PR-Reviewer-R14-Review-Body-Linter.md`

## Mechanism

The workflow already has a single explicit pytest command for PR-review tooling.
This PR appends `tests/test_check_review_body_r14.py` to that command so the
checker's 19 behavioral tests run in CI alongside the workflow wiring tests.

The merged #1472 plan is moved to `plans/archive/`, and
`scripts/archive_plans.py index` refreshes `plans/INDEX.md` so the plan catalog
matches the file layout.

## Intentional

- No checker logic changes. The review note is about CI enrollment only; changing
  matcher behavior again would expand the slice.
- No new workflow job. The linter tests belong in the existing PR-review tooling
  unit-test list because that is where the checker and workflow tests are
  already exercised.

## Deferred

- None.

Parked hardening: none.

## Verification

- Focused R14 linter suite
  - Pytest for `tests/test_check_review_body_r14.py` passed, 19 tests.
- PR-review tooling unit-test list from `.github/workflows/pre_push_audit.yml`
  - `python -m pytest tests/test_local_pr_review.py tests/test_audit_ai_reconciliation.py tests/test_audit_review_rules_triggered.py tests/test_summarize_review_misses.py tests/test_check_ai_reconciliation_live.py tests/test_pre_push_audit_workflow.py tests/test_check_review_body_r14.py -q` passed, 82 tests.
- Plan archive index refresh
  - `python scripts/archive_plans.py index` passed.
- Plan and diff audits
  - `python scripts/sync_pr_plan.py plans/PR-Reviewer-R14-Linter-CI-Enrollment.md origin/main --check` passed.
  - `scripts/audit_plan_code_consistency.py` passed for this plan.
  - `scripts/audit_review_rules_triggered.py` passed for this plan.
  - `git diff --name-status origin/main...HEAD` confirmed only the intended
    four files after rebasing onto current `origin/main`.
  - `git diff --check origin/main...HEAD` passed.

## Estimated diff size

| File | LOC |
|---|---:|
| `.github/workflows/pre_push_audit.yml` | 2 |
| `plans/INDEX.md` | 3 |
| `plans/PR-Reviewer-R14-Linter-CI-Enrollment.md` | 95 |
| `plans/archive/PR-Reviewer-R14-Review-Body-Linter.md` | 0 |
| **Total** | **100** |

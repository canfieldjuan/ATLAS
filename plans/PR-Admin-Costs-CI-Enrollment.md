# PR: Admin Costs CI Enrollment

## Why this slice exists

The #1043 review confirmed the full `tests/test_admin_costs.py` suite is green
after the saved-calls fix, but also noted that host CI did not run that file.
That is how the pre-existing red tests from #1040 stayed hidden.

This slice makes the green admin-costs suite load-bearing in GitHub Actions,
without expanding the global pre-push audit for every PR.

## Scope (this PR)

Ownership lane: content-ops/cost-surfacing
Slice phase: Workflow/process

1. Add a path-scoped GitHub Actions workflow for the admin costs API surface.
2. Run `tests/test_admin_costs.py` when the workflow, admin costs API, or admin
   costs tests change.
3. Install the repo runtime requirements before running the route-level test.
4. Leave the broader pre-push audit and extracted pipeline workflows unchanged.

### Files touched

| File | Change |
|---|---|
| `plans/PR-Admin-Costs-CI-Enrollment.md` | Plan doc for the admin-costs CI enrollment slice. |
| `.github/workflows/admin_costs_checks.yml` | Add a path-scoped workflow for the admin costs test suite. |

## Mechanism

The workflow runs on pull requests and pushes to `main` when these paths change:
the workflow file itself, `atlas_brain/api/admin_costs.py`, or
`tests/test_admin_costs.py`. It checks out the repo, sets up Python 3.11,
enables pip caching, installs the root runtime requirements, installs pytest
and pytest-asyncio, and runs:

```bash
python -m pytest tests/test_admin_costs.py -q
```

Installing root requirements matches the repo-precedented host-touching
workflows. The admin-costs module imports through the aggregate API package, so
the workflow should load the real host dependencies instead of relying on a
hand-picked subset or test-level import shims.

## Intentional

- This uses a dedicated path-scoped workflow instead of adding the test to
  `pre_push_audit.yml`, because admin-costs route tests should gate relevant
  admin-costs changes without slowing unrelated PRs.
- This does not enroll the entire host test suite. The reviewer finding was
  specific to `tests/test_admin_costs.py`.
- This installs root requirements instead of maintaining a fragile curated
  dependency subset for a host route test.
- This does not touch product code or mutate test import state.

## Deferred

- Future PR: add similar path-scoped workflows for other host route suites if
  they are found green-but-ungated.
- Parked hardening: none. Root `HARDENING.md` was scanned; this directly closes
  the #1043 reviewer forward note.

## Verification

- Workflow YAML parse command: python inline PyYAML load for `.github/workflows/admin_costs_checks.yml` — passed.
- Admin costs pytest command: python -m pytest tests/test_admin_costs.py -q — 21 passed, 1 warning.
- First GitHub Actions run — failed during collection because the workflow did
  not install `asyncpg`, which is imported by `atlas_brain.storage.database`.
- Second GitHub Actions run — failed during collection on an unrelated
  aggregate API import path, `atlas_brain/api/__init__.py` -> speaker repository
  -> missing `numpy`.
- Clean temporary venv probe after adding `numpy` and `dateparser` exposed the
  next unrelated aggregate import failure at missing `torch`, confirming that
  dependency-chasing would be patchwork for this route-level workflow.
- Reviewer repro showed the test-level import shim broke a combined pytest
  session with `tests/test_task_trigger_reasons.py`; the shim was removed and
  the workflow now installs root requirements instead.
- Combined regression command: python -m pytest tests/test_admin_costs.py
  tests/test_task_trigger_reasons.py -q — 29 passed, 1 warning.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan doc | ~85 |
| Workflow | ~45 |
| **Total** | **~130** |

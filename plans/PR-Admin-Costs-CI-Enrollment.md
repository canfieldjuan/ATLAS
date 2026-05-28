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
3. Install only the dependencies needed for this route-level test file.
4. Keep the focused route test from importing the aggregate API router package,
   which pulls unrelated app dependencies into this path-scoped workflow.
5. Leave the broader pre-push audit and extracted pipeline workflows unchanged.

### Files touched

| File | Change |
|---|---|
| `plans/PR-Admin-Costs-CI-Enrollment.md` | Plan doc for the admin-costs CI enrollment slice. |
| `.github/workflows/admin_costs_checks.yml` | Add a path-scoped workflow for the admin costs test suite. |
| `tests/test_admin_costs.py` | Isolate the route import from aggregate API package side effects. |

## Mechanism

The workflow runs on pull requests and pushes to `main` when these paths change:
the workflow file itself, `atlas_brain/api/admin_costs.py`, or
`tests/test_admin_costs.py`. It checks out the repo, sets up Python 3.11,
installs the same lightweight route-test dependencies used by adjacent CI jobs
plus `asyncpg` and `psutil`, and runs:

```bash
python -m pytest tests/test_admin_costs.py -q
```

The test module registers a lightweight `atlas_brain.api` package shim with the
real API package path before importing `atlas_brain.api.admin_costs`. That lets
the route test import the admin-costs module directly without executing
`atlas_brain/api/__init__.py`, whose aggregate router imports unrelated app
surfaces and heavy dependencies that this route test does not exercise. It uses
the same package-shim pattern for `atlas_brain.services` and the scraping
package, then supplies a tiny parser registry stub for the parser-version values
the admin-costs assertions need. That keeps the admin route behavior under test
without importing every scraper implementation.

## Intentional

- This uses a dedicated path-scoped workflow instead of adding the test to
  `pre_push_audit.yml`, because admin-costs route tests should gate relevant
  admin-costs changes without slowing unrelated PRs.
- This does not enroll the entire host test suite. The reviewer finding was
  specific to `tests/test_admin_costs.py`.
- This changes the route test import instead of installing the full host
  requirements or unrelated heavy dependencies in CI. The failing dependency
  chain came from package import side effects, not from the admin-costs route
  behavior under test.
- The parser registry is stubbed only for parser version values used by this
  admin-costs suite. Parser implementation coverage belongs in scraper tests.
- This does not touch product code or the production aggregate API router.

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
- Clean temporary venv with the workflow dependency list after test import
  isolation: python -m pytest tests/test_admin_costs.py -q — 21 passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan doc | ~95 |
| Workflow | ~45 |
| Test import isolation | ~30 |
| **Total** | **~170** |

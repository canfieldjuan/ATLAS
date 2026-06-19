# PR-CI-Repo-Wide-Unit-Backstop

## Why this slice exists

CI runs no repo-wide test suite. Every `*_checks.yml` workflow is path-filtered
and executes an explicit, hand-maintained list of test files, and the one safety
net (`scripts/audit_extracted_pipeline_ci_enrollment.py`) only audits files whose
names match a fixed allowlist (`test_extracted_content*`, `test_extracted_ticket_faq*`,
...). A test file named outside those patterns -- e.g. the recently added
`tests/test_deflection_snapshot_report_drift.py` -- can therefore pass all of CI
while never running until someone manually enrolls it. That is a silent-coverage
hole: a brand-new test, or a regression in a file enrolled only under another
lane's path filter, can ship green.

This slice adds the missing catch-all: a scheduled, on-demand full unit run so
nothing hides indefinitely. It does not touch the fast path-filtered per-PR
checks, which stay as the quick first line.

## Scope (this PR)

Ownership lane: ci/coverage
Slice phase: Production hardening

1. Add `.github/workflows/repo_wide_unit_backstop.yml` running
   `pytest -m "not integration and not e2e"` over the whole `tests/` tree
   (integration/e2e need Postgres/Neo4j and are out of scope for a no-services
   runner).
2. Triggers: `schedule` (daily, single commented cron with a UTC<->Central
   note), `workflow_dispatch` for on-demand, and a `pull_request` trigger
   **scoped to this workflow file only** so the backstop is exercised on changes
   to itself without becoming a slow gate on every unrelated PR.
3. `permissions: contents: read`, a `concurrency` group, and a per-job
   `timeout-minutes` cap, matching repo workflow conventions.

### Files touched

- `.github/workflows/repo_wide_unit_backstop.yml`
- `plans/PR-CI-Repo-Wide-Unit-Backstop.md`

### Review Contract

Acceptance criteria:

- [ ] The workflow runs the unit suite repo-wide (no per-file allowlist) and
      excludes integration/e2e markers.
- [ ] It is not a per-PR gate: only schedule, manual dispatch, and self-file
      changes trigger it.
- [ ] Actions and permissions follow the repo's workflow-security posture
      (`scripts/audit_workflow_security_posture.py` passes).

Affected surfaces: CI only; no application code.

Risk areas: a long full-suite runtime; surfacing pre-existing failures that the
path-filtered lanes never ran (that is the intended signal, not a regression in
this change).

Reviewer rules triggered: R1, R14.

## Mechanism

A single job installs `requirements.txt` + pytest and runs
`python -m pytest -m "not integration and not e2e" -q`. Because it does not
path-filter the test set, a file that no per-area workflow enrolls is still
executed here. The `pull_request` path filter is the workflow file itself, which
makes the backstop self-exercising on this PR (immediate proof it runs) while
keeping it off the critical path of normal PRs.

## Intentional

- Not a per-PR gate. The per-area path-filtered checks remain the fast layer;
  this is the nightly/on-demand backstop, so it does not slow every PR.
- No change to the enrollment auditor in this slice. Broadening the auditor's
  name-pattern allowlist (so non-conforming files are flagged at PR time) is a
  separate slice because that auditor is scoped to one workflow + runner and
  would red-fail until cross-workflow enrollment is taught to it.
- Integration/e2e excluded; those need Postgres/Neo4j and belong to their own
  service-backed jobs.

## Deferred

- Broaden `audit_extracted_pipeline_ci_enrollment.py` (and make it
  workflow-aware) so deflection/content-ops test families are flagged for
  enrollment at PR time, not only caught nightly by this backstop.
- Optionally gate merges on a green backstop once its runtime/flake profile is
  known.

Parked hardening: none.

## Verification

- `python -c "import yaml; yaml.safe_load(open('.github/workflows/repo_wide_unit_backstop.yml'))"`
  -- valid.
- `python scripts/audit_workflow_security_posture.py` -- workflow security
  posture audit passed.
- The backstop runs on this PR via the self-scoped `pull_request` trigger; its
  CI result is the live proof of repo-wide execution.

## Estimated diff size

| File | LOC |
|---|---:|
| `.github/workflows/repo_wide_unit_backstop.yml` | 52 |
| `plans/PR-CI-Repo-Wide-Unit-Backstop.md` | 95 |
| **Total** | **147** |

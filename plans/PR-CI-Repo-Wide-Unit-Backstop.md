# PR-CI-Repo-Wide-Unit-Backstop

## Why this slice exists

CI runs no repo-wide test suite. Every `*_checks.yml` workflow is path-filtered
and executes an explicit, hand-maintained list of test files, so a test file no
workflow enrolls can pass all of CI while never running. The earlier attempt at
a catch-all (closed PR #1707) could not go green because the full suite could
not be collected; the collection gaps are now fixed by the merged slices in
`docs/backstop_test_hygiene_scope.md` (invoicing enrollment #1710, external
test collection fixes #1711, and the mcp/asyncpg isolation slices).

This slice does two coupled things:

1. **Backstop (E):** add the scheduled/on-demand repo-wide unit run so nothing
   hides indefinitely.
2. **Auditor backstop-awareness (G):** teach
   `scripts/audit_extracted_pipeline_ci_enrollment.py` that the repo-wide
   backstop is a valid catch-all, so a touched `atlas_brain`-importing test no
   longer requires a bespoke `atlas_*_checks.yml` once the backstop guarantees
   coverage. Without G, the auditor would block every PR that touches a
   previously-unenrolled `atlas_brain` test (e.g. the remaining mcp/asyncpg
   isolation slices) even though the backstop already runs them.

## Scope (this PR)

Ownership lane: ci/coverage
Slice phase: Production hardening

1. Add `.github/workflows/repo_wide_unit_backstop.yml` running
   `pytest -m "not integration and not e2e"` over the whole `tests/` tree,
   triggered by `schedule` + `workflow_dispatch` + a `pull_request` filter
   scoped to the workflow file itself.
2. Add `repo_wide_backstop_present()` to
   `scripts/audit_extracted_pipeline_ci_enrollment.py` and short-circuit
   `atlas_brain_test_workflow_errors()` when the backstop exists.
3. Add a backstop-aware acceptance test to
   `tests/test_audit_extracted_pipeline_ci_enrollment.py`.

### Files touched

- `.github/workflows/repo_wide_unit_backstop.yml`
- `scripts/audit_extracted_pipeline_ci_enrollment.py`
- `tests/test_audit_extracted_pipeline_ci_enrollment.py`
- `plans/PR-CI-Repo-Wide-Unit-Backstop.md`

### Review Contract

Acceptance criteria:

- [ ] The workflow runs the unit suite repo-wide (no per-file allowlist) and
      excludes `integration`/`e2e`.
- [ ] It is not a per-PR gate: only schedule, manual dispatch, and self-file
      changes trigger it.
- [ ] The auditor credits backstop coverage: a changed `atlas_brain`-importing
      test passes when the backstop exists, and still fails when it does not
      (existing failure tests unchanged).

Affected surfaces: CI + the CI-enrollment auditor; no application code.

Risk areas: relaxing the auditor's per-file requirement -- mitigated by gating
the relaxation on the backstop workflow actually existing and running the
repo-wide unit suite.

Reviewer rules triggered: R1, R2, R10, R14.

## Mechanism

The workflow installs `requirements.txt` + pytest and runs
`pytest -m "not integration and not e2e" -q --continue-on-collection-errors`.
Because it does not path-filter the test set, a file no per-area workflow
enrolls is still executed here.

`repo_wide_backstop_present()` parses the backstop workflow's YAML `run:` steps
and returns true only when one invokes `pytest -m "not integration and not
e2e"` over the whole tree (no per-file target) -- so marker strings lingering
in a comment/echo, or a path-limited command, do not falsely credit coverage.
`atlas_brain_test_workflow_errors()` then decides per changed test:

- An `integration`/`e2e`-marked test is **exempt** -- it is service-lane, the
  backstop skips it, and this unit-enrollment audit does not cover it.
- A unit test is treated as enrolled when the backstop runs it; otherwise the
  existing per-file dedicated-enrollment logic applies unchanged.

## Intentional

- Backstop is advisory (not a per-PR merge gate) for its debut, so its
  runtime/flake profile is observed before it gates anything. The fast
  path-filtered per-PR checks are untouched.
- `--continue-on-collection-errors` stays as belt-and-suspenders: one
  un-importable optional-dependency module cannot zero out the whole run's
  signal.
- The auditor relaxation is gated on a real repo-wide pytest run step (parsed
  from the workflow YAML), not raw-text substrings.
- Crediting the backstop relaxes **PR-time** per-file gating for unit tests in
  favor of the scheduled/on-demand catch-all: a touched unit test runs in the
  nightly backstop rather than on the PR. This is the deliberate tradeoff of
  the backstop-aware approach; per-area path-filtered checks remain the fast
  PR-time layer for the files they already cover.
- Integration/e2e atlas_brain tests are exempted rather than credited to the
  backstop (which never runs them), so the auditor makes no false coverage
  claim for them.

## Deferred

- The backstop's first run is expected red until the residual
  `setdefault("asyncpg", MagicMock())` planters (~37 files) are isolated, AND
  the unmarked live/DB tests (e.g. `tests/test_evidence_claim_builder_live.py`,
  which opens a real asyncpg pool but marks only `asyncio`) are marked or
  excluded so the scheduled run stays unit-only. Both are tracked follow-ups in
  `docs/backstop_test_hygiene_scope.md`, not blockers for an advisory check.
- Optionally gate merges on a green backstop once its runtime/flake profile is
  known.

Parked hardening: none.

## Verification

- `python -c "import yaml; yaml.safe_load(open('.github/workflows/repo_wide_unit_backstop.yml'))"`
  -- valid.
- `scripts/audit_workflow_security_posture.py` -- workflow security posture
  audit passes.
- `python -m pytest tests/test_audit_extracted_pipeline_ci_enrollment.py -q`
  -- 21 passed (adds backstop-accept, comment-only-reject, and
  integration-exempt cases).

## Estimated diff size

| File | LOC |
|---|---:|
| `.github/workflows/repo_wide_unit_backstop.yml` | 54 |
| `scripts/audit_extracted_pipeline_ci_enrollment.py` | ~60 |
| `tests/test_audit_extracted_pipeline_ci_enrollment.py` | ~75 |
| `plans/PR-CI-Repo-Wide-Unit-Backstop.md` | ~165 |
| **Total** | **~354** |

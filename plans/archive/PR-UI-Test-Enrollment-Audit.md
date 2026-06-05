# PR-UI-Test-Enrollment-Audit

## Why this slice exists

PR #1320 closed the immediate hole from #1318 (9 atlas-intel-ui `test:*` suites
that CI never ran) by hand-wiring the missing steps. But the run list in each
`*_ui_checks.yml` is still hand-maintained, so the drift can silently recur — which
is exactly how it happened the first time (and, per AGENTS.md, "has been dropped
four times"). This slice makes the hole un-droppable: a mechanical audit that fails
when any declared UI `test:*` script is not run by its CI workflow. This is PR-B of
the #1318 plan.

The diff is ~450 LOC, over the 400 soft cap. The overage is review-driven
robustness (anchoring the parser to `run:` step bodies and surfacing a malformed
`package.json` as drift, each with its own negative fixture) rather than added
scope — splitting a single auditor and its fixtures across two PRs would ship a
weaker checker first, which defeats the slice's purpose.

## Scope (this PR)

Ownership lane: intel-ui/ci-enrollment
Slice phase: Workflow/process

1. Add `scripts/audit_ui_test_enrollment.py`: for each `*-ui` package, every
   `test:*` script must be invoked by its conventional checks workflow
   (directory `foo-ui` maps to its foo_ui_checks.yml workflow) via an explicit
   `npm run test:<name>` step.
   Fails on a declared-but-unrun test, and on a UI that declares tests but has no
   checks workflow at all.
2. Ship `tests/test_audit_ui_test_enrollment.py` with §3h coverage.
3. Wire the audit into `scripts/pre_push_audit.sh` so it runs in local review and
   in the `pre-push-audit` CI workflow.
4. Enroll the new fixture test in CI beside the existing CI-tooling audit fixture.

### Files touched

- `.github/workflows/extracted_pipeline_checks.yml`
- `plans/PR-UI-Test-Enrollment-Audit.md`
- `scripts/audit_ui_test_enrollment.py`
- `scripts/pre_push_audit.sh`
- `scripts/run_extracted_pipeline_checks.sh`
- `tests/test_audit_ui_test_enrollment.py`

## Mechanism

The audit auto-discovers UI packages (`*-ui` dirs that contain a package.json) and maps
each to its workflow by convention (`workflow_name_for`), so a newly added UI with
tests is covered automatically rather than having to be remembered — the audit is
not itself a hand-maintained list. `parse_test_scripts` reads the declared
`test:*` keys; `parse_workflow_runs` extracts `npm run test:<name>` tokens via
regex; `audit_root` reports per-UI rows (`OK` / `NO_TESTS` / `MISSING_WORKFLOW` /
`UNENROLLED`) and `main` exits non-zero on drift. Declared-vs-run is a set
difference, so a longer run-step name cannot satisfy a shorter declared name by
prefix.

Against the current tree the audit is clean: atlas-intel-ui (24) and portfolio-ui
(3) fully enrolled; the other four `*-ui` packages declare no `test:*` scripts.

## Intentional

- **UI -> workflow is derived by convention, not a hardcoded list.** A hardcoded
  mapping would reintroduce the very hand-maintained-list drift this audit exists to
  prevent.
- **A UI with no `test:*` scripts is reported `NO_TESTS`, not drift.** No tests means
  no enrollment drift is possible; the row is still printed (surfaced, not silently
  skipped) per AGENTS.md §3g.
- **`atlas-mobile` is out of scope.** It is React Native with a distinct test
  runner and is not a `*-ui` package; it declares no `test:*` today. If it gains
  test scripts, extend `discover_ui_dirs` (noted in Deferred).
- **The fixture test is enrolled in `run_extracted_pipeline_checks.sh`**, beside
  `test_audit_extracted_pipeline_ci_enrollment.py` — the existing home for
  CI-tooling audit fixtures in CI. Shipping a fixture CI never runs would repeat the
  exact sin this PR fixes.

## Deferred

- Reverse drift (a workflow `npm run test:<name>` whose script does not exist) is
  not audited: `npm run` of a missing script already fails CI, so it is
  self-detecting.
- Extending discovery beyond `*-ui` (e.g. `atlas-mobile`) if/when it adds a test
  runner.
- Broader audit-fixture enrollment: most `tests/test_audit_*.py` are not run by any
  CI workflow today; enrolling that whole suite is its own slice.
- Parked hardening: none.

## Verification

```bash
python scripts/audit_ui_test_enrollment.py        # exit 0; intel-ui + portfolio-ui OK
python -m pytest tests/test_audit_ui_test_enrollment.py -q   # 9 passed
python scripts/audit_extracted_pipeline_ci_enrollment.py     # still OK after enrolling the fixture
bash scripts/local_pr_review.sh --current-pr-body-file tmp/pr-body-ui-test-enrollment-audit.md
```

Verified locally: audit exit 0, 9/9 fixtures pass, CI-enrollment audit still OK
(148 enrolled).

## Estimated diff size

| File | LOC |
|---|---:|
| `.github/workflows/extracted_pipeline_checks.yml` | 4 |
| `plans/PR-UI-Test-Enrollment-Audit.md` | 108 |
| `scripts/audit_ui_test_enrollment.py` | 185 |
| `scripts/pre_push_audit.sh` | 1 |
| `scripts/run_extracted_pipeline_checks.sh` | 1 |
| `tests/test_audit_ui_test_enrollment.py` | 157 |
| **Total** | **456** |

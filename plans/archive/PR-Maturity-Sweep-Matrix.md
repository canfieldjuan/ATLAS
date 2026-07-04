# PR-Maturity-Sweep-Matrix

## Why this slice exists

Issue #1962 tracks the Atlas CI/CD map and safe speedup work. The runtime audit
in `docs/ci_cd_runtime_duplication_audit.md` names broad `Maturity Sweep`
runtime as the highest-ranked safe optimization candidate: the workflow runs
many independent ratchet groups serially, so PR wall time trends toward the sum
of every group instead of the slowest group.

Root cause: independent maturity-sweep ratchets are sequenced inside one
blocking job even though they do not share mutable state. This change fixes the
orchestration root by moving those blocking ratchet groups into matrix legs
while preserving the unit tests, advisory sweep, ratchet commands, baselines,
sensitive-path arguments, and a single review-visible aggregate pass condition.

This branch also archives the merged #1982 plan doc as required teardown
housekeeping. The #1982 content is preserved in its squash commit; keeping it in
the root `plans/` directory would make an already-merged slice look in flight.

This slice exceeds the 400 LOC soft budget because the safe implementation
keeps all 15 existing ratchet command groups intact while moving them from
serial workflow steps into matrix rows. Splitting the mechanical rewrite across
multiple PRs would leave the workflow half-shaped and harder to verify.

## Scope (this PR)

Ownership lane: workflow/autonomous-ci-cd-map
Slice phase: Workflow/process

1. Split the broad `Maturity Sweep` workflow into a unit/advisory job, a
   blocking ratchet matrix job, and a small aggregate `maturity-sweep` job.
2. Preserve every existing ratchet command group and baseline, only changing
   orchestration from serial to parallel matrix legs.
3. Add a workflow-shape regression test proving the unit/advisory coverage,
   matrix groups, ratchet markers, and aggregate gate stay wired.
4. Archive the merged #1982 plan doc and refresh the plan index.

### Review Contract

- Acceptance criteria:
  - [ ] The existing sweep unit tests and non-blocking advisory sweep still run.
  - [ ] Every current blocking ratchet command group remains present with the
        same lane/baseline/sensitive-path markers.
  - [ ] Independent ratchet groups run as matrix legs with `fail-fast: false`
        so one failure does not hide other ratchet failures.
  - [ ] A stable aggregate `maturity-sweep` job fails unless both the
        unit/advisory job and the matrix ratchet job succeed.
  - [ ] A committed test proves the workflow coverage shape so future edits
        cannot silently drop a ratchet or the aggregate gate.
  - [ ] The merged #1982 plan is archived from root `plans/`.
- Reachability proof: `python -m pytest` runs the workflow-shape test and the
  existing maturity sweep workflow tests; `scripts/local_pr_review.sh`
  exercises the plan/workflow mechanical gate.
- Affected surfaces: CI workflow orchestration, workflow-shape tests, plan
  archive metadata.
- Risk areas: CI enrollment, false-green ratchet coverage, branch-protection
  context churn, runtime performance.
- Reviewer rules triggered: R1, R2, R12, R14.

### Files touched

- `.github/workflows/maturity_sweep_advisory.yml`
- `plans/INDEX.md`
- `plans/PR-Maturity-Sweep-Matrix.md`
- `plans/archive/PR-Session-Bootstrap-Temporal-Discipline.md`
- `tests/test_maturity_sweep_advisory_workflow.py`

## Mechanism

`.github/workflows/maturity_sweep_advisory.yml` keeps the existing workflow triggers. The old
single `maturity-sweep` job becomes three jobs:

1. `maturity-sweep-unit-and-advisory` runs checkout, Python setup, pytest for
   the sweep/tooling tests, and the existing advisory sweep with
   `continue-on-error: true`.
2. `maturity-sweep-ratchets` uses a `strategy.matrix.include` list where each
   matrix row owns one current ratchet group and its exact shell command. The
   matrix uses `fail-fast: false` to report all failing groups in one run.
3. `maturity-sweep` remains as the stable aggregate job. It uses `needs` and
   `if: always()` so it runs after both upstream jobs and exits non-zero if
   either upstream job failed, was cancelled, or was skipped.

The new test reads the workflow file as text and asserts the matrix and
aggregate invariants directly. That keeps the equivalence proof close to the
CI surface without adding a YAML parser dependency.

## Intentional

- No ratchet, threshold, baseline, or sensitive glob is removed. This slice
  changes scheduling shape only.
- The aggregate job keeps a stable review-visible pass condition instead of
  asking operators to mentally combine many matrix statuses.
- This does not narrow path filters. The audit explicitly deferred that for
  Maturity Sweep because `scripts/maturity_sweep.py` indexes the full tests
  tree via `--tests-root tests`.
- This does not change pre-push audit, live reconciliation, secret scanning,
  diff budget, or PR body contract gates.

## Deferred

- Product workflow dependency-cache/install improvements remain a later #1962
  speedup candidate.
- Extracted Pipeline decomposition remains a later package-ownership slice.
- Path-filter narrowing remains deferred until each workflow's dependency on
  `tests/**` is proven safe to narrow.

Parked hardening: none.

## Verification

- `python -m pytest tests/test_maturity_sweep_advisory_workflow.py tests/test_retired_failure_detector_workflow.py --noconftest -q` - 7 passed.
- `python -m pytest tests/test_maturity_sweep.py tests/test_detect_retired_failure_modes.py tests/test_maturity_sweep_advisory_workflow.py tests/test_retired_failure_detector_workflow.py --noconftest -q` - 41 passed.
- `python - <<'PY' ... yaml.safe_load(.github/workflows/maturity_sweep_advisory.yml) ... PY` - parsed.
- `python scripts/maturity_sweep.py extracted_content_pipeline --tests-root tests --baseline tests/maturity_sweep/baseline_extracted_content_pipeline.json --min-score 8 --sensitive-glob '**/billing/**' --sensitive-glob '**/paid*' --sensitive-glob '**/auth/**' --sensitive-glob '**/webhook*' --sensitive-glob '**/payment*' --sensitive-glob '**/*deletion*'` - ratchet gate passed.
- `python scripts/maturity_sweep.py scripts --tests-root tests --baseline tests/maturity_sweep/baseline_scripts.json --min-score 8 --sensitive-glob 'scripts/**'` - ratchet gate passed.
- `python scripts/sync_pr_plan.py plans/PR-Maturity-Sweep-Matrix.md --check` - passed.
- `git diff --check` - passed.
- `ATLAS_SESSION_STATE_FILE=SESSION_STATE.maturity-sweep-matrix-1962.local.md scripts/local_pr_review.sh --allow-dirty` via bash - passed as a pre-commit advisory pass.

## Estimated diff size

| File | LOC |
|---|---:|
| `.github/workflows/maturity_sweep_advisory.yml` | 561 |
| `plans/INDEX.md` | 3 |
| `plans/PR-Maturity-Sweep-Matrix.md` | 130 |
| `plans/archive/PR-Session-Bootstrap-Temporal-Discipline.md` | 0 |
| `tests/test_maturity_sweep_advisory_workflow.py` | 132 |
| **Total** | **826** |

# PR-Local-Unit-Gate-Mirror

## Why this slice exists

The operator observed repeated PRs going red on the branch-required `unit-gate`
after publication and asked whether the gate can run before opening the PR. The
current wrappers already route publish/open through `scripts/local_pr_review.sh`,
but that local bundle does not execute the same selector/check path as
`.github/workflows/unit_gate.yml`, so GitHub becomes the first place a unit-gate
failure is discovered.

### Problem-derived contract

- Root cause: Local PR publication runs the pre-push/local-review mechanics but
  omits the branch-required unit-gate selection and ratchet check, so a builder
  can open or update a PR that has already-failing required unit-gate evidence.
- Correct fix must touch/change: `scripts/local_pr_review.sh` must run a local
  mirror of the unit-gate selector/check flow before reporting success, and
  `tests/test_local_pr_review.py` must prove the mirror runs locally, fails
  closed, strips wrapper-only PR body/Git hook env from the unit-gate
  subprocess, propagates selector failures before dispatching the checker,
  covers all three selector outcomes, and does not duplicate the separate
  GitHub Actions `unit_gate` job.
- Must not change: Do not alter `.github/workflows/unit_gate.yml`,
  `scripts/check_unit_gate.py`, `scripts/select_impacted_tests.py`, the unit
  baseline, product code, product shape, or review verdict policy.

## Scope (this PR)

Ownership lane: atlas-workflow/pr-enforcement
Slice phase: Workflow/process

1. Add a late-stage `Local unit gate mirror` step to `scripts/local_pr_review.sh`.
2. Add focused local-review tests for selected-test execution, empty-selection
   growth-only execution, `FULL` execution, selector failure propagation, and
   GitHub Actions duplication avoidance.

### Review Contract

- Acceptance criteria:
  - `scripts/local_pr_review.sh` invokes the local unit-gate mirror before
    printing `local PR review passed` when earlier local checks have no failures.
  - The mirror resolves the base unit-gate baseline from the merge base with the
    supplied base ref, writes an empty baseline when the base has no ledger, and
    then runs the same `check_unit_gate.py` modes as the workflow: `FULL`,
    `--growth-only`, or `--selected-files` with the selected pytest paths.
  - A failing local mirror increments the local-review failure count and prevents
    the wrappers from opening/updating a PR as green.
  - A failing selector exits the mirror before any `check_unit_gate.py` dispatch,
    and the failure increments the local-review failure count.
  - The mirror unsets wrapper-only PR body environment and Git hook-local
    environment before running selector and unit-gate subprocesses, matching the
    standalone GitHub `unit_gate` job.
  - Under `GITHUB_ACTIONS=true`, `local_pr_review.sh` skips the mirror because
    `.github/workflows/unit_gate.yml` is already a separate required CI job.
- Reachability proof: Real entrypoint is `bash scripts/local_pr_review.sh`; the
  observable effect is the `Local unit gate mirror` section running or skipping
  and the final local-review exit code.
- Affected surfaces: `scripts/local_pr_review.sh` and
  `tests/test_local_pr_review.py`.
- Risk areas: local/CI duplication, trusted script-root execution, expensive
  full-suite selection, baseline-ratchet parity.
- Reviewer rules triggered: R1, R2, R6, R7, R10, R12, R14.

### Boundary-change enumeration

Required when this diff changes a guard, validator, normalizer, resolver,
router/classifier, or admission boundary. Name each changed boundary path or
seam in the enumeration; otherwise write "N/A - no boundary change."

- Boundary path/seam: `scripts/local_pr_review.sh` success admission now depends
  on the local mirror of the branch-required unit gate.
- Replaced-path behaviors: Previously local review could pass without any unit
  gate evidence; now it runs the mirror unless earlier local checks already
  failed, the check script is absent, or GitHub Actions is already running the
  dedicated unit-gate workflow.
- Guard-relevant fields: `base_ref`, `repo_root`, `script_root`,
  `GITHUB_ACTIONS`, `ATLAS_CURRENT_PR_BODY_FILE`, `ATLAS_CURRENT_PR_AUTHOR`,
  Git hook-local variables from `git rev-parse --local-env-vars`, selected test
  file contents, and the merge-base baseline.
- Caller x input shape: `scripts/push_pr.sh` and `scripts/open_pr.sh` call
  `local_pr_review.sh --current-pr-body-file ...`; the pre-push hook calls
  `local_pr_review.sh`; GitHub `pre_push_audit` calls the same script with
  `--repo-root` and `--script-root`.

### Deployed-config probing

Required for guard, validator, resolver, admission-boundary, or env/config
fallback changes; otherwise write "N/A - no guard/config boundary change."

- Deployed/default config values: Default local shell has `GITHUB_ACTIONS`
  unset, so the mirror runs when `scripts/check_unit_gate.py` exists and earlier
  local checks passed.
- Explicit value probe: `GITHUB_ACTIONS=true` skips the mirror, leaving the
  dedicated CI `unit_gate` job as the required remote authority.
- Absent value probe: If `scripts/check_unit_gate.py` is absent, local review
  prints a skip instead of pretending to run the gate.
- Default-session/default-context probe: Normal `push_pr.sh` / `open_pr.sh`
  local publication inherits the unset `GITHUB_ACTIONS` default and therefore
  runs the mirror.
- Side-effect ordering: The mirror runs after cheap local-review checks and
  before the final success message; if earlier checks failed, it skips to avoid
  spending full-suite time on an already-invalid PR.

### Files touched

- `plans/PR-Local-Unit-Gate-Mirror.md`
- `scripts/local_pr_review.sh`
- `tests/test_local_pr_review.py`

## Mechanism

`scripts/local_pr_review.sh` gains `run_local_unit_gate_mirror`, which creates
temporary selection and base-baseline files, resolves the merge-base baseline
using the same base-ref contract as the workflow, runs
`scripts/select_impacted_tests.py --base <base-ref>` when the selector exists in
the PR tree, and dispatches to `scripts/check_unit_gate.py` in the same three
modes as `.github/workflows/unit_gate.yml`. The selector/check subprocesses run
without `ATLAS_CURRENT_PR_BODY_FILE`, `ATLAS_CURRENT_PR_AUTHOR`, or Git
hook-local variables so the local mirror matches the standalone CI unit-gate
environment instead of the publication wrapper environment.

Selector execution and checker dispatch record and return child exit codes
explicitly; the function does not depend on Bash `errexit` because
`run_check` invokes checks through an `if` condition.

The call is deliberately late in local review. Cheap plan/body/drift/diff checks
run first; if any failed, the mirror reports a skip because the PR is already
blocked. When `GITHUB_ACTIONS=true`, the mirror also reports a skip so the
trusted-base pre-push audit does not duplicate the dedicated required
`unit_gate` workflow.

## Intentional

- No change to the workflow, selector, unit-gate checker, or baseline; this PR
  only moves discovery earlier in the local publish path.
- The mirror can run the full unit suite locally when the selector escalates to
  `FULL`. That cost is intentional because it is the exact red check the
  operator asked to catch before opening or updating the PR.
- The mirror skips in GitHub Actions to avoid a second copy of the same
  branch-required unit gate inside `pre_push_audit`.

## Deferred

- Unit-gate runtime optimization or caching, if the full-suite path remains too
  slow after failures move local.
- Additional selector ownership refinements as separate narrow PRs when a
  concrete changed path escalates unnecessarily.

Parked hardening: none.

## Verification

- `python -m pytest tests/test_local_pr_review.py -q` - 25 passed.
- `GITHUB_ACTIONS=true python -m pytest tests/test_local_pr_review.py -q` - 25 passed.
- `bash -n scripts/local_pr_review.sh` - passed.
- Scoped CI-mode unit gate invocation for `tests/test_local_pr_review.py` -
  passed with zero regressions against the merge-base baseline.
- `bash scripts/local_pr_review.sh --current-pr-body-file /tmp/atlas-pr-local-unit-gate-mirror-body.md`
  - passed, including `Local unit gate mirror` selecting
    `tests/test_local_pr_review.py` and reporting zero regressions.

## Estimated diff size

| File | LOC |
|---|---:|
| `plans/PR-Local-Unit-Gate-Mirror.md` | 170 |
| `scripts/local_pr_review.sh` | 93 |
| `tests/test_local_pr_review.py` | 126 |
| **Total** | **389** |

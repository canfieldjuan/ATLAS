# PR-Unit-Gate

Ownership lane: ci-cd/enforcement-gaps

Phase 1 / slice 2 of the CI/CD enforcement gap tracker (#2035, gap G1.1 /
root cause RC1). Builds on slice 1 (#2036, pinned deps).

## Why this slice exists

The CI/CD audit (#2035) found no PR-required check executes any test (RC1):
every code-executing workflow is path-filtered, so a change to an unlisted
module (`atlas_brain/voice/**`, `services/llm_router.py`, `asr_server.py`,
`graphiti-wrapper/**` -- G2) merges with zero test execution. The one workflow
that runs the whole suite, `.github/workflows/repo_wide_unit_backstop.yml`, is structurally
incapable of gating: schedule-only, and its lone `pull_request` trigger is
neutered (`continue-on-error` on PR at `:51`, path-filtered to its own file at
`:20-22`). It cannot simply be flipped to blocking: the repo-wide unit suite
has 185 deterministic pre-existing failures (177 failed + 8 error, verified
byte-identical across 3 nightly runs 07-05..07-07), which would red-wall every
PR.

This slice adds a per-PR gate that runs the whole unit suite and blocks only on
a REGRESSION beyond a committed baseline (ratchet), tolerating the 185 known
failures while catching any new one. It lands after slice 1 because the ratchet
baseline is only meaningful on a reproducible (pinned) dependency set.

Diff-budget note: over 400 LOC because `tests/unit_gate_baseline.txt` is a
190-line DATA ledger (the 188 known-failing node ids), not logic. The logic is
~250 LOC (checker + tests). The ledger may only shrink as failures are
remediated.

### Problem-derived contract

Root cause: no PR-required check runs tests, and the whole-suite workflow is
built to never gate. A correct fix runs the whole unit suite on every PR and
fails only on a failing node NOT already in a committed baseline; it does not
fix or hide the 185 pre-existing failures, does not touch the nightly backstop,
and does not (yet) change branch protection.

## Scope (this PR)

Slice phase: workflow/process

- `scripts/check_unit_gate.py` -- runs the backstop's exact pytest invocation,
  parses the failing/errored node set, fails iff any failing node is not in the
  baseline (regression); reports baseline entries that now pass (ratchet-shrink).
- `tests/unit_gate_baseline.txt` -- the committed 188-node known-failures
  ledger (185 nightly + 3 pull_request-context), reconciled to CI on this PR.
- `.github/workflows/unit_gate.yml` -- `pull_request` (all paths) +
  `workflow_dispatch`; installs pinned deps; invokes the checker. Not added to
  required checks this slice (advisory at birth; enrolling it is slice 3).
- `tests/test_check_unit_gate.py` -- gate-logic unit tests (no 30-min suite).
- This plan doc.

Nothing else. No test-file edits, no backstop edit, no branch-protection change.

### Files touched

- `.github/workflows/unit_gate.yml`
- `plans/PR-Unit-Gate.md`
- `scripts/check_unit_gate.py`
- `tests/test_check_unit_gate.py`
- `tests/unit_gate_baseline.txt`

### Review Contract

Acceptance criteria (check one-by-one):
1. The gate runs `pytest tests/ -m "not integration and not e2e"
   --continue-on-collection-errors` (the backstop's invocation) on every PR,
   all paths, and its verdict is the checker's own exit code -- no
   `continue-on-error`/`|| true` neutering.
2. Checker exits 1 on a failing node absent from the baseline (regression),
   0 when the failing set is a subset of the baseline, 2 on usage/IO error --
   proven by `tests/test_check_unit_gate.py` (both directions).
3. The 185 pre-existing failures are recorded, not fixed or hidden; no
   test-file, `pytest.ini`, marker, or conftest change.
4. `.github/workflows/repo_wide_unit_backstop.yml` and branch-protection required checks are
   untouched; the gate is advisory at birth.

Reachability proof: the gate's `pull_request` trigger runs it on THIS PR --
observable output = the `unit-gate` check on #<pr> executing the full suite and
reporting green (CI failing set subset of the baseline; the baseline is
reconciled to CI on that run). A seeded new failure turns it red, proven by
`test_cli_exit1_on_regression`.

Affected surfaces: a new PR check `unit-gate`; a new maintenance script; a new
committed baseline ledger. No runtime code, no existing workflow, no gating
policy.

Risk areas: baseline drift between the seed (nightly, unpinned) and the gate's
pinned-dep CI run (mitigated: reconciled on this PR's run before merge); a
flaky test entering the suite would make the ratchet noisy (mitigated for now:
the current 185 are verified deterministic; flakes get baselined + tracked).

Reviewer rules triggered: R10, R2 (evaluator/gate predicate -- the checker,
boundary-probed both directions); R11/R12 do not apply (no env/config schema).

## Mechanism

`check_unit_gate.py` runs pytest via subprocess with `-rfE --tb=no -q
--continue-on-collection-errors`, parses `FAILED `/`ERROR ` summary lines into a
node-id set (non-greedy up to an optional ` - <message>`, so parametrized ids
with spaces stay whole), loads the baseline (ignoring `#`/blank lines), and
computes `regressions = failing - baseline` and `fixed = baseline - failing`.
Exit 1 iff regressions. pytest's expected non-zero (baseline failures exist) is
captured, not propagated -- the gate verdict is the script's exit, decided by
the set comparison. `--update-baseline` regenerates the ledger; `--report-file`
gates a pre-captured summary (used by the tests to avoid the real suite).

Integrity guards (added in review):
- **Ran-check:** `ensure_pytest_ran` fails the gate (exit 2) when pytest's exit
  status is not 0/1 -- a usage/plugin/internal error produces no summary lines
  and would otherwise parse to an empty failing set and go silently green.
- **Ratchet growth guard:** `--base-baseline` rejects (exit 3) any node the PR
  adds to the ledger vs the base branch -- a PR cannot baseline its own new
  failure to pass. An empty/absent base file = the initial seed (this PR).
- **Param-safe parse:** the node-id regex captures an optional `[...]` group
  whole, so parametrized ids with spaces/dashes are not truncated.
- **Pinned test deps:** the workflow installs `pytest==9.1.1
  pytest-asyncio==1.4.0` (the versions that produced the baseline), so an
  unpinned pytest release cannot drift the ledger.
- **Exact match, not superset:** the baseline must EQUAL the failing set --
  both a new failure (regression) and a stale entry that now passes fail the
  gate. A superset ledger is a live allow-list hole (a later PR could
  reintroduce a baselined failure and still pass), so the ratchet enforces
  shrinkage, not just advises it.
- **Head-determinism:** the gate checks out the PR HEAD (not the default
  merge-with-base ref), so concurrent `main` advances cannot leak new failures
  into the run and make the baseline a moving target. Merge-time breakage with
  current `main` is covered by the cross-session drift auditor, not this gate.

## Intentional

- **Whole-suite, not changed-file subset.** Closing the G2 blind spot requires
  running tests for changed-but-unlisted modules; a changed-file heuristic
  reintroduces the "which tests cover this file" gap. Cost: a ~15-30 min full
  run per PR, in parallel with the fast path-filtered checks.
- **Ratchet, not all-must-pass.** 185 deterministic pre-existing failures make
  all-must-pass a 185-test fix (separate work). The baseline tolerates them and
  can only shrink.
- **Advisory at birth.** The check reports genuinely (real exit code) but is
  not yet required -- enrolling it (branch-protection change) is slice 3, after
  it is proven stable on live PRs.
- **Separate from the backstop.** The nightly `repo_wide_unit_backstop` stays
  the full-failure signal; this gate is the per-PR ratchet sibling.

## Deferred

- Enroll `unit-gate` in `required_status_checks` -- slice 3 (branch-protection
  change, its own reviewed step).
- Remediate the 185 baselined failures and shrink the ledger -- ongoing,
  separate from the gate.
- Speed (changed-file-aware selection, split shards) -- revisit at slice 3 if
  the full run is too slow to require.
- Flaky-test policy (reruns / quarantine marker) -- only if a flake enters the
  suite; the current 185 are deterministic.
- Nested `[]` inside a parametrized id (vanishingly rare) still truncates at
  the inner `]`; a switch to structured pytest output (`--report-log`) would
  remove all regex fragility -- deferred, no current baseline id is affected.
- The 3 baselined `test_audit_pr_session_drift` cli tests fail only in the
  `pull_request` CI context (they pass locally and in the `schedule` nightly);
  investigating/fixing that context-sensitivity is separate remediation.

## Verification

- Local: `pytest tests/test_check_unit_gate.py -q` green (9 tests, gate logic
  both directions); `scripts/check_ascii_python.sh`-clean; the committed
  baseline parses, is sorted + unique (188).
- CI (this PR): the `unit-gate` check runs the full suite on all paths and
  reports green -- the reconciliation of the baseline to the pinned-dep CI
  failing set happens on that run; any delta is reconciled before merge.

## Estimated diff size

| File | LOC |
|---|---:|
| `.github/workflows/unit_gate.yml` | 69 |
| `plans/PR-Unit-Gate.md` | 182 |
| `scripts/check_unit_gate.py` | 223 |
| `tests/test_check_unit_gate.py` | 165 |
| `tests/unit_gate_baseline.txt` | 194 |
| **Total** | **833** |

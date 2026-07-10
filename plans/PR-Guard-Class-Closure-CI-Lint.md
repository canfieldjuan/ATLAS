# PR-Guard-Class-Closure-CI-Lint

## Why this slice exists

`PR-Guard-Class-Closure` (#2066, merged) codified the class-closure discipline
as prose in `docs/GUARD_CLASS_CLOSURE.md` + `AGENTS.md` section 3k.1 +
`docs/REVIEWER_RULES.md`. Prose rules get followed inconsistently -- the whole
S6A arc (9+ rounds) is what happens when the rule exists only in a reviewer's
head. That PR named a CI lint as the deferred enforcement follow-up. This slice
builds it: an advisory check that surfaces a warning when a PR changes a guard
over an open input space without a co-changed property/generative test, so the
omission is visible on every PR instead of depending on a reviewer catching it.

Advisory-first by deliberate choice (the same rollout the CI-enforcement arc
uses): "guard-shaped over open input" cannot be detected precisely, so a
required gate would false-positive and wedge unrelated PRs. The lint warns and
exits 0; a `--strict` mode and a required enrollment are the named future step
after an advisory-proving period.

Diff-budget override: the ~568-line diff exceeds the 400-line soft target
because a CI gate is one indivisible unit -- the detector, its both-direction
tests, the workflow enrollment, and the plan ship together or the gate is
uninstalled/unproven. The runtime detector itself is ~236 lines; tests (138)
and the plan (140) are most of the remainder.

### Problem-derived contract

- Root cause: the class-closure bar is documentation only; nothing makes a
  guard-shaped PR that ships a fixture list (not a property test) visible in
  CI, so the rule depends entirely on reviewer memory.
- Correct fix must touch/change: a checker that (1) detects guard-shaped source
  changes over open input heuristically, (2) detects whether the same diff adds
  a property/generative test, and (3) reports advisory warnings; its own
  both-direction tests; an advisory (non-required) CI workflow; a discoverable
  opt-out (config + inline waiver); and a one-line pointer from the rule doc.
- Must not change: no product/runtime code, no existing required check, no
  branch protection. The lint is non-blocking and cannot wedge a merge.

## Scope (this PR)

Ownership lane: process/review-discipline
Slice phase: Workflow/process

Max files: 5

1. Add `scripts/check_guard_class_closure.py` -- pure detection core
   (`scan_diff`) plus a thin git transport; advisory by default, `--strict`
   for a future required enrollment; opt-out via config + inline waiver.
2. Add `tests/test_check_guard_class_closure.py` -- both-direction fixtures:
   guard change without a property test is flagged; with a property test, a
   non-guard change, and the single-signal near-misses are clean.
3. Add `.github/workflows/guard_class_closure.yml` -- advisory job that runs
   the detector tests and the lint; reads only the diff (no PR-code execution).
4. Add the advisory-lint pointer to `docs/GUARD_CLASS_CLOSURE.md`.

### Review Contract

- Acceptance criteria:
  - [ ] The detection core is pure (`scan_diff(added_by_file)`), so tests
        exercise the real logic with synthetic diffs; only git is mocked.
  - [ ] A guard-shaped source change (path-name stem, OR both a verdict def and
        an open-input signal) with no co-changed property test is flagged.
  - [ ] The same change WITH a co-changed property/generative test
        (parametrize / itertools.product / hypothesis) is clean; a plain
        fixture list is NOT treated as a property test (still flagged).
  - [ ] Single-signal near-misses (verdict-only, open-input-only) and non-guard
        changes are NOT flagged (false-positive control).
  - [ ] The lint is advisory: exits 0 on findings (warnings), non-zero only
        under `--strict`; the CI job is non-required and cannot block a merge.
  - [ ] Opt-out works: an `ignore_globs` config entry and an inline
        `guard-class-closure: waived` marker both suppress a finding.
  - [ ] The workflow reads only the diff/file contents (no execution of PR
        code) and passes the workflow-security-posture audit.
- Reachability proof: run the detector on a synthetic guard diff (in tests) and
  smoke the installed CLI against `origin/main`; assert the advisory output and
  exit code.
- Affected surfaces: developer tooling, CI enrollment (advisory).
- Risk areas: false positives (mitigated: advisory-only + opt-out + both
  content signals required), heuristic guard detection, PR-ref execution
  (mitigated: read-only diff).
- Reviewer rules triggered: R1, R2, R10, R12, R14.

### Files touched

- `.github/workflows/guard_class_closure.yml`
- `docs/GUARD_CLASS_CLOSURE.md`
- `plans/PR-Guard-Class-Closure-CI-Lint.md`
- `scripts/check_guard_class_closure.py`
- `tests/test_check_guard_class_closure.py`

## Mechanism

`scripts/check_guard_class_closure.py` splits into a pure core and a thin git
layer. `scan_diff({path: added-lines})` classifies each changed non-test .py
file as guard-shaped (a guard path-name stem, OR both a verdict-def signal and
an open-input signal in the added lines) and, when any guard file changed with
no co-changed property/generative test in the same diff, returns advisory
findings. The git layer (`changed_added_lines`) shells `git diff` to build that
map; tests bypass it and call the pure core with synthetic diffs, so the real
detection logic is exercised and only the transport is mocked. Default output
is GitHub `::warning::` annotations with exit 0; `--strict` exits non-zero for a
future required enrollment. The workflow runs the detector tests and the lint on
`pull_request`, reading only the diff.

## Intentional

- Advisory-first and non-required: a heuristic guard detector must not wedge
  unrelated PRs. Promotion to required (`--strict` + branch protection) is the
  named future step after it proves itself, matching the CI-enforcement arc.
- Both content signals (verdict AND open-input) are required for a content-based
  match, trading some recall for a low false-positive rate appropriate to an
  advisory gate; path-name stems are a high-precision shortcut.
- A plain fixture list is deliberately NOT counted as a property test -- that is
  the exact anti-pattern the rule targets.

## Deferred

- Promote to a required check with `--strict` after an advisory-proving period
  (operator policy flip; same shape as the unit-gate enrollment).
- A semantic check that the property test asserts against a spec oracle (not
  just parity) -- the lint can see a property test exists but not that its
  oracle is independent; that stays a reviewer judgment for now.

Parked hardening: none.

## Verification

- Detector tests: run tests/test_check_guard_class_closure.py (13 cases, both
  directions) -- green.
- Dogfood smoke: run the installed CLI against origin/main on this branch and
  confirm it does not flag its own diff (the checker ships a property test).
- ASCII gate (check_ascii_python.sh) -- green (the new .py files are ASCII).
- Workflow security posture audit (audit_workflow_security_posture.py) --
  green for the new workflow.
- Plan sync (sync_pr_plan.py --check) -- in sync.

## Estimated diff size

| File | LOC |
|---|---:|
| `.github/workflows/guard_class_closure.yml` | 46 |
| `docs/GUARD_CLASS_CLOSURE.md` | 8 |
| `plans/PR-Guard-Class-Closure-CI-Lint.md` | 146 |
| `scripts/check_guard_class_closure.py` | 236 |
| `tests/test_check_guard_class_closure.py` | 138 |
| **Total** | **574** |

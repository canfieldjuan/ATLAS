# PR-Guard-Class-Closure-Claude-Contract

## Why this slice exists

The operator reported that `scripts/check_guard_class_closure.py` already exists
but is not being followed by Claude Code sessions, so review-fix PRs can still
patch the reviewer-cited symptom instead of closing the defect class. The
checker is visible in an advisory GitHub workflow, but the read-first Claude
contract does not name the exact command or the stop rule, and local PR review
does not run the checker before push.

This workflow/process slice makes the existing guard-class closure rule
preventative for builders: CLAUDE.md must tell Claude when to run the checker
and what a finding means, and `local_pr_review.sh` must run the checker in
strict mode so an unclosed guard-shaped diff stops before push unless the
existing waiver marker is present.

Diff-budget override: the local-gate wiring exposed coupled review failures on
the same guard-class closure boundary. The trusted-root execution fix,
trusted-policy-root separation, strict detector strengthening, selector closure
declaration, and red-path tests must land together so the new local stop gate is
both enforced and truthful.

### Problem-derived contract

- Root cause: the repository has a guard-class closure detector, but the
  builder-facing contract and local review bundle do not mechanically require
  Claude Code sessions to use it before pushing review-fix or guard-shaped
  changes. The result is a gap between the intended R13 discipline and the
  commands builders actually run.
- Correct fix must touch/change: add command-level CLAUDE.md guidance for
  `scripts/check_guard_class_closure.py`; run that checker from
  `scripts/local_pr_review.sh` with `--strict`; make the checker inspect the PR
  worktree under trusted-base local review while loading ignore policy only from
  the trusted script checkout; strengthen strict mode so existing guard body
  edits, bare generative-looking syntax, fixture matrices, string-scoped product
  costumes, or unused Hypothesis imports do not satisfy class closure; enroll the
  checker/local-review surfaces in impacted-test selection; add tests proving
  the strict local gate, waiver path, trusted-root path, and CLAUDE.md command
  contract.
- Must not change: do not rewrite the canonical guard-closure bar in
  `docs/GUARD_CLASS_CLOSURE.md`; do not promote the remote GitHub workflow to
  branch-required/trusted-base; do not change product behavior or
  reviewer-rule semantics beyond making the existing builder gate visible and
  strict locally.

## Scope (this PR)

Ownership lane: workflow/guard-class-closure-claude-contract
Slice phase: Workflow/process
Max files: 9

1. Make Claude Code guidance name the guard-class closure checker, the command
   to run, and the meaning of a finding.
2. Make local PR review run `scripts/check_guard_class_closure.py --strict` so
   guard-shaped diffs without class-level proof or waiver block before push.
3. Make strict mode reject weak same-example parametrized fixture lists and
   string-scoped product costumes while still allowing generative
   product/Hypothesis evidence.
4. Add focused tests and impacted-test enrollment for the local gate and
   CLAUDE.md contract.

### Review Contract

- Acceptance criteria:
  1. `CLAUDE.md` names `scripts/check_guard_class_closure.py`, gives the exact
     `python scripts/check_guard_class_closure.py --base origin/main --strict`
     command, and says a finding blocks another symptom patch unless the PR
     adds class-level proof or carries the existing `guard-class-closure:
     waived` marker with rationale.
  2. `scripts/local_pr_review.sh` runs the guard-class closure checker with
     `--strict` after the PR body is available and before the local unit mirror.
  3. The checker honors `ATLAS_AUDIT_REPO_ROOT`, so trusted-base local review
     inspects the materialized PR worktree rather than the trusted script
     checkout.
4. Strict mode rejects a bare `@pytest.mark.parametrize` fixture list,
     literal-only `product(...)` matrices, string-scoped product costumes, and
     unused Hypothesis imports, while accepting generative `itertools.product` /
     Hypothesis evidence tied to the guard module with grammar axes and an
     independent oracle marker.
  5. A synthetic local-review fixture with a guard-shaped diff and no
     property/generative test fails local review.
  6. The same synthetic fixture passes when the PR body carries the existing
     guard-class closure waiver marker.
  7. Impacted-test selection maps `scripts/check_guard_class_closure.py` and
     the CLAUDE.md contract check to focused tests.
- Reachability proof: run `bash scripts/local_pr_review.sh` on this PR and the
  focused pytest files; the observable effect is the local review output
  including a strict guard-class closure gate before push.
- Affected surfaces: `CLAUDE.md`, `scripts/check_guard_class_closure.py`,
  `scripts/local_pr_review.sh`, `scripts/select_impacted_tests.py`, guard
  checker tests, local-review tests, selector tests, and a CLAUDE contract test.
- Risk areas: false-positive guard blocking, waiver handling, command drift in
  CLAUDE.md, selector under-selection, and accidental remote workflow promotion.
- Reviewer rules triggered: R1, R2, R10, R12, R13, R14.

### Boundary-change enumeration

Required when this diff changes a guard, validator, normalizer, resolver,
router/classifier, or admission boundary. Name each changed boundary path or
seam in the enumeration; otherwise write "N/A - no boundary change."

- Boundary path/seam: local PR review admission for guard-shaped diffs.
- Replaced-path behaviors: before this slice, `local_pr_review.sh` did not run
  `scripts/check_guard_class_closure.py`; after this slice, it runs the checker
  with `--strict` and fails on findings unless the existing waiver marker is in
  the PR body.
- Guard-relevant fields: changed Python files, added diff lines, co-changed
  test files, `ATLAS_CURRENT_PR_BODY_FILE`, `guard-class-closure: waived`,
  trusted ignore config, base ref, and `ATLAS_AUDIT_REPO_ROOT`.
- Caller x input shape: `bash scripts/local_pr_review.sh --current-pr-body-file <body> [base-ref]`
  before push/open.

### Deployed-config probing

Required for guard, validator, resolver, admission-boundary, or env/config
fallback changes; otherwise write "N/A - no guard/config boundary change."

- Deployed/default config values: N/A - local builder review gate, no deployed
  configuration.
- Explicit value probe: tests provide a PR body containing
  `guard-class-closure: waived`, and tests prove PR-side ignore-config data
  cannot override trusted ignore policy.
- Absent value probe: tests provide the same guard-shaped diff with no waiver
  and no property/generative test.
- Default-session/default-context probe: `bash scripts/local_pr_review.sh` runs
  the checker with its default `origin/main` base unless a base ref is supplied.
- Side-effect ordering: guard-class closure runs before the local unit-gate
  mirror so symptom patching blocks before expensive tests.

### Closure declaration

- Set-valued dependency: `scripts/select_impacted_tests.py`'s
  `EXPLICIT_TEST_OWNERS` entries added by this slice.
- Is the set closed or open? CLOSED for this slice: the membership is exactly
  the two non-import-graph surfaces this PR changes and must unit-enroll
  (`CLAUDE.md` and `scripts/check_guard_class_closure.py`).
- Where does membership come from? ENUMERATED/AUTHORED HERE in the selector's
  explicit owner map because these surfaces are not discoverable by the Python
  import graph. The owning tests are the contract tests this PR adds or extends:
  `tests/test_claude_guard_class_contract.py`,
  `tests/test_check_guard_class_closure.py`, `tests/test_local_pr_review.py`,
  and `tests/test_select_impacted_tests.py`.
- What happens outside the set? Existing selector behavior remains unchanged:
  unknown Python/global CI surfaces escalate to `FULL`, while regular
  Markdown-only docs remain provably test-free unless explicitly mapped. This
  slice explicitly maps `CLAUDE.md` because it is a read-first agent contract,
  not ordinary prose.

### Files touched

- `CLAUDE.md`
- `plans/PR-Guard-Class-Closure-Claude-Contract.md`
- `scripts/check_guard_class_closure.py`
- `scripts/local_pr_review.sh`
- `scripts/select_impacted_tests.py`
- `tests/test_check_guard_class_closure.py`
- `tests/test_claude_guard_class_contract.py`
- `tests/test_local_pr_review.py`
- `tests/test_select_impacted_tests.py`

## Mechanism

Add a strict guard-class closure stage to `local_pr_review.sh`. It reuses the
existing checker and passes the same PR body via `ATLAS_CURRENT_PR_BODY_FILE`
that the checker already reads for waivers. The checker now separates its
trusted policy root from the inspected Git root: `ATLAS_AUDIT_REPO_ROOT` points
git diff commands at the PR worktree, while the optional guard-class closure
ignore policy is loaded from the trusted script checkout when that config
exists.

Strengthen strict mode inside `scripts/check_guard_class_closure.py`: advisory
mode still surfaces weak `@pytest.mark.parametrize` signals, but strict mode
requires generator syntax, explicit grammar axes, an independent
oracle/expected-verdict marker, a tie to the guard module, strict verdict-hunk
detection for existing guard body edits, and non-literal axis fixtures. That
prevents a single cited-example parametrized list, unused Hypothesis import,
literal-only `product(...)` matrix, or string-scoped product costume from
satisfying the new local stop gate.

Add CLAUDE.md guidance in the PR workflow/root-cause area so Claude Code sees
the exact command and the disposition rule before it starts another review-fix
push. Add focused tests that prove the local review stage fails on an unclosed
guard-shaped diff, passes with the existing waiver marker, and keeps CLAUDE.md
from drifting back to vague prose.

## Intentional

- Keep the remote GitHub workflow advisory in this slice; trusted-base/required
  promotion is a separate security and branch-protection change.
- Reuse the existing waiver marker rather than creating a second disposition
  vocabulary.
- Keep advisory mode's heuristic scope; only strict mode gets the stronger
  property/generative evidence bar.
- The cross-layer caller hint on `tests/test_local_pr_review.py`'s private
  `_write_fixture_repo` helper is test-only reuse. The helper's existing
  behavior is preserved; it now also installs the guard checker so local-review
  fixtures exercise the same gate this PR adds.
- Cross-layer hints on `scan_diff` and `main` are same-name functions in
  separate checker modules, not call sites of `scripts/check_guard_class_closure.py`;
  the real guard-checker callers are its CLI and tests covered in this slice.

## Deferred

- Trusted-base remote promotion for `guard-class-closure-lint` after the
  operator chooses to make the GitHub check required.
- Broader set-valued dependency detection from
  `docs/GUARD_CLASS_CLOSURE.md` trigger B.

Parked hardening: none.

## Verification

- `python -m pytest tests/test_claude_guard_class_contract.py -q` - passed, 2
  tests.
- `python -m pytest tests/test_check_guard_class_closure.py -q` - passed, 30
  tests.
- `python -m pytest tests/test_local_pr_review.py -q` - passed, 31 tests.
- `python -m pytest tests/test_select_impacted_tests.py tests/test_claude_guard_class_contract.py -q`
  - passed, 68 tests.
- `python -m pytest tests/test_check_guard_class_closure.py tests/test_local_pr_review.py tests/test_select_impacted_tests.py tests/test_claude_guard_class_contract.py -q`
  - passed, 129 tests.
- `python scripts/check_guard_class_closure.py --base origin/main --strict` -
  passed.
- Pending before push: refreshed plan/body audits and local PR review.

## Estimated diff size

| File | LOC |
|---|---:|
| `CLAUDE.md` | 14 |
| `plans/PR-Guard-Class-Closure-Claude-Contract.md` | 240 |
| `scripts/check_guard_class_closure.py` | 93 |
| `scripts/local_pr_review.sh` | 15 |
| `scripts/select_impacted_tests.py` | 8 |
| `tests/test_check_guard_class_closure.py` | 133 |
| `tests/test_claude_guard_class_contract.py` | 23 |
| `tests/test_local_pr_review.py` | 100 |
| `tests/test_select_impacted_tests.py` | 9 |
| **Total** | **635** |

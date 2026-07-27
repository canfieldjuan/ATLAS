# PR-Plan-Code-Consistency-Deleted-Paths

## Why this slice exists

PR #2240 intentionally hard-deletes retired review-gate files. GitHub
`pre-push-audit` runs trusted `origin/main` scripts against the PR checkout, so
it uses the base version of `scripts/audit_plan_code_consistency.py`. That base
checker treats deleted files listed in the plan as missing path claims and also
treats full backticked commands as path claims when the command ends in `.md` or
`.py`. This precursor lands the checker fix first so the broader Codex review
scope reset can be judged by the corrected trusted-base gate.

### Problem-derived contract

- Root cause: `scripts/audit_plan_code_consistency.py` assumes every
  enforceable backticked path claim must exist on disk. That is wrong for a PR
  whose plan truthfully lists a file being deleted by the branch. The same token
  parser also mistakes command strings containing spaces for path claims.
- Correct fix must touch/change: update
  `scripts/audit_plan_code_consistency.py` so command strings are not path
  tokens, branch-deleted paths resolve as valid plan claims against the selected
  review base, and local review passes that selected base into the checker; add
  focused tests in `tests/test_audit_plan_code_consistency.py` and
  `tests/test_local_pr_review.py`.
- Must not change: product behavior, customer-visible surfaces, plan admission,
  diff-budget, `live-reconciliation`, PR ownership checks, or the broader Codex
  review-scope reset in #2240.

## Scope (this PR)

Ownership lane: workflow/plan-code-consistency
Slice phase: Workflow/process

Max files: 5

1. Narrow plan/code consistency to stop false positives on branch-deleted path
   claims and command-string tokens.
2. Pass local review's selected base ref into the plan/code checker.
3. Add fixture coverage for the false-positive and selected-base cases.

### Review Contract

- Acceptance criteria:
  - `parse_claims` does not classify a backticked command containing spaces as
    a path claim, including path-headed executable commands with no flags.
  - `parse_claims` still preserves literal path claims that contain spaces.
  - `audit_claims` accepts a path claim when the path is deleted in the current
    branch diff, including names with pathspec metacharacters and non-ASCII
    basenames.
  - Deleted-path resolution uses the caller-selected base ref rather than a
    hard-coded `origin/main`.
  - Local review passes its selected base ref into plan/code consistency.
  - Existing path/function-claim behavior remains covered by the existing
    checker tests.
- Reachability proof: N/A - this is a local/CI workflow checker with pytest
  fixture coverage, not a runtime product surface.
- Affected surfaces: `scripts/audit_plan_code_consistency.py`,
  `scripts/local_pr_review.sh`, `tests/test_audit_plan_code_consistency.py`,
  and `tests/test_local_pr_review.py`.
- Risk areas: checker false negatives, command-token parsing, deleted-file diff
  detection, basename fallback for deleted paths, selected-base propagation,
  success-output accuracy, and CI/local parity.
- Reviewer rules triggered: R1, R2, R10, R12, R14.

### Boundary-change enumeration

Required when this diff changes a guard, validator, normalizer, resolver,
router/classifier, or admission boundary. Name each changed boundary path or
seam in the enumeration; otherwise write "N/A - no boundary change."

- Boundary path/seam: `scripts/audit_plan_code_consistency.py` validates
  backticked path/function claims in plan docs.
- Replaced-path behaviors: missing path claims still fail unless the path exists,
  is gitignored session state, or is deleted by the current branch diff.
- Guard-relevant fields: backticked tokens, shell-split executable shape,
  whitespace inside tokens, path suffixes, basename-only claims, and
  NUL-terminated `git diff --diff-filter=D` deleted-path entries.
- Caller x input shape: local review and CI pass a plan doc path plus selected
  base ref; pytest calls `parse_claims` and `audit_claims` directly.

### Deployed-config probing

Required for guard, validator, resolver, admission-boundary, or env/config
fallback changes; otherwise write "N/A - no guard/config boundary change."

- Deployed/default config values: N/A - no deployed config or fallback behavior.
- Explicit value probe: N/A.
- Absent value probe: N/A.
- Default-session/default-context probe: N/A.
- Side-effect ordering: N/A.

### Files touched

- `plans/PR-Plan-Code-Consistency-Deleted-Paths.md`
- `scripts/audit_plan_code_consistency.py`
- `scripts/local_pr_review.sh`
- `tests/test_audit_plan_code_consistency.py`
- `tests/test_local_pr_review.py`

## Mechanism

The checker keeps its existing path/function claim audit. The token predicate
now excludes command-shaped backticked strings by shell-splitting the token,
skipping environment assignments, and classifying the executable shape instead
of matching command-name or marker allowlists; literal path claims with spaces
still remain path claims. Path resolution now has a second successful case after
an on-disk lookup: deleted-path discovery reads `git diff --name-only -z
--diff-filter=D` and compares full path claims exactly or basename-only claims
by basename. In both cases the plan claim is valid even though the file no
longer exists in the checkout. Local review forwards its selected base ref to
the checker so PRs targeting a non-`main` base compare deletions against the
same base used by the rest of the review bundle.

## Intentional

- Keep this as a precursor rather than burying it in #2240; GitHub's trusted-base
  CI cannot use a checker fix that only exists inside the PR being checked.
- Do not broaden the checker into a full Markdown command parser; whitespace is
  not enough by itself because literal repo paths can contain spaces, so this
  only filters command-shaped tokens.

## Deferred

- Rebase #2240 on this precursor after it lands, then rerun the broad Codex
  review-scope reset with the corrected trusted-base checker.

Parking predicate: park only follow-up hardening that does not affect whether
this checker correctly classifies command-shaped tokens, branch-deleted path
claims, basename shorthand, or caller-selected base refs.

Parked hardening: none.

## Verification

- python -m pytest tests/test_audit_plan_code_consistency.py tests/test_local_pr_review.py -q - 29 passed.
- python scripts/audit_plan_code_consistency.py --base-ref origin/main plans/PR-Plan-Code-Consistency-Deleted-Paths.md - passed.
- python scripts/maturity_sweep.py scripts --tests-root tests --baseline tests/maturity_sweep/baseline_scripts.json --min-score 8 --sensitive-glob 'scripts/**' - passed; ratchet gate reported no new brittleness above baseline.

## Estimated diff size

| File | LOC |
|---|---:|
| `plans/PR-Plan-Code-Consistency-Deleted-Paths.md` | 148 |
| `scripts/audit_plan_code_consistency.py` | 103 |
| `scripts/local_pr_review.sh` | 4 |
| `tests/test_audit_plan_code_consistency.py` | 110 |
| `tests/test_local_pr_review.py` | 23 |
| **Total** | **388** |

# PR-Plan-Code-Consistency-Deleted-Paths

## Why this slice exists

The EOM lead-review queue vertical (#2242) archives prior plan docs as part of
normal slice hygiene, and the follow-on review-scope cleanup (#2240) hard-deletes
retired review-gate files. Trusted-base GitHub `pre-push-audit` still runs
`origin/main`'s plan/code checker, so a checker that misreads branch-deleted or
renamed path claims and command-shaped backticks can block operator-visible EOM
funnel PRs before their own fixed checker is trusted. This precursor fixes that
trusted gate before #2240 and later EOM funnel slices depend on it.

Diff-budget overage rationale: this slice is genuinely indivisible because the
checker mechanism, selected-base forwarding, CI enrollment, regression coverage,
and plan contract must move together; splitting any one of those parts leaves
the trusted plan/code gate either unable to run the new coverage or still
false-red/false-green for EOM funnel PRs.

### Problem-derived contract

- Root cause: `scripts/audit_plan_code_consistency.py` requires every
  enforceable path claim to exist on disk and mistakes command strings for path
  claims.
- Root-cause disposition: this change fixes that root cause for the declared
  command/path and branch-deleted/renamed claim classes; it is not merely a
  symptom waiver or PR-body exception.
- Correct fix must touch/change: update
  `scripts/audit_plan_code_consistency.py` so command strings are not path
  tokens, branch-deleted and branch-renamed source paths resolve as valid plan
  claims against the selected review base, and local review passes that selected
  base into the checker; add focused tests in
  `tests/test_audit_plan_code_consistency.py` and `tests/test_local_pr_review.py`;
  enroll the checker tests in `.github/workflows/pre_push_audit.yml`.
- Must not change: product behavior, customer-visible surfaces, plan admission,
  diff-budget, `live-reconciliation`, PR ownership checks, or the broader Codex
  review-scope reset in #2240.

## Scope (this PR)

Ownership lane: workflow/plan-code-consistency
Slice phase: Workflow/process

Max files: 6

1. Narrow plan/code consistency to stop false positives on branch-deleted path
   claims and command-string tokens.
2. Pass local review's selected base ref into the plan/code checker.
3. Add fixture coverage for the false-positive and selected-base cases.

### Review Contract

- Acceptance criteria:
  - `parse_claims` does not classify a backticked command containing spaces as
    a path claim, including extensionless path-headed executables and
    path-headed executable commands with no flags such as "./tools/run
    tests/example.py".
  - `parse_claims` still preserves literal path claims that contain spaces,
    including basename shorthand such as "ATLAS Distributed System.txt".
  - `audit_claims` accepts a path claim when the path is deleted or renamed away
    in the current branch diff, including names with pathspec metacharacters,
    non-ASCII basenames, basename shorthand, and repository-relative spellings
    with a leading `./`.
  - Deleted-path resolution uses the caller-selected base ref rather than a
    hard-coded `origin/main`.
  - Local review passes its selected base ref into plan/code consistency.
  - Existing path/function-claim behavior remains covered by the existing
    checker tests.
- Reachability proof: N/A - this is a local/CI workflow checker with pytest
  fixture coverage, not a runtime product surface.
- Affected surfaces: `.github/workflows/pre_push_audit.yml`,
  `scripts/audit_plan_code_consistency.py`,
  `scripts/local_pr_review.sh`, `tests/test_audit_plan_code_consistency.py`,
  and `tests/test_local_pr_review.py`.
- Risk areas: checker false negatives, command-token parsing, deleted-file diff
  detection, basename fallback for deleted paths, selected-base propagation,
  success-output accuracy, and CI/local parity.
- Reviewer rules triggered: R1, R2, R10, R12, R14.

### Boundary-change enumeration

- Boundary path/seam: `scripts/audit_plan_code_consistency.py` validates
  backticked path/function claims in plan docs; `.github/workflows/pre_push_audit.yml`
  enrolls the checker regression tests in trusted-base CI.
- Replaced-path behaviors: missing path claims still fail unless the path exists,
  is gitignored session state, or is deleted by the current branch diff.
- Guard-relevant fields: backticked tokens, shell-split executable shape,
  whitespace inside tokens, path suffixes, basename-only claims, and
  normalized repository-relative claims from NUL-terminated `git diff
  --diff-filter=DR --find-renames` deleted-path and renamed-source entries.
- Caller x input shape: local review and CI pass a plan doc path plus selected
  base ref; pytest calls `parse_claims` and `audit_claims` directly.

### Deployed-config probing

- Deployed/default config values: N/A - no deployed config or fallback behavior.
- Explicit value probe: N/A.
- Absent value probe: N/A.
- Default-session/default-context probe: N/A.
- Side-effect ordering: N/A.

### Files touched

- `.github/workflows/pre_push_audit.yml`
- `plans/PR-Plan-Code-Consistency-Deleted-Paths.md`
- `scripts/audit_plan_code_consistency.py`
- `scripts/local_pr_review.sh`
- `tests/test_audit_plan_code_consistency.py`
- `tests/test_local_pr_review.py`

## Mechanism

The checker still audits path/function claims, but shell-splits backticked tokens
to classify only the declared command set: known command heads, executable
script suffixes, and path-headed executable invocations. Unknown whitespace
tokens that look like repository path claims remain path claims, which preserves
literal paths and basename shorthand with spaces. Deleted-path discovery reads
`git diff --name-status -z
--diff-filter=DR --find-renames`, normalizes repository-relative `./` spellings,
and accepts exact full-path or basename-only deleted and renamed-source claims.
Local review forwards its base ref so non-`main` PR targets compare deletions
against the same base as the rest of the review bundle. The trusted-base
pre-push workflow runs the checker regression tests alongside the rest of the
PR-review tooling tests.

## Intentional

- Keep this as a precursor rather than burying it in #2240; GitHub's trusted-base
  CI cannot use a checker fix that only exists inside the PR being checked.
- Do not broaden the checker into a full Markdown command parser; whitespace is
  not enough by itself because literal repo paths can contain spaces. The
  default is therefore path-claim preserving unless the token matches the
  declared command set.

## Deferred

- Rebase #2240 on this precursor after it lands, then rerun the broad Codex
  review-scope reset with the corrected trusted-base checker.

Parking predicate: park only follow-up hardening that does not affect whether
this checker correctly classifies command-shaped tokens, branch-deleted path
claims, basename shorthand, or caller-selected base refs.

Parked hardening: none.

## Verification

- python -m pytest tests/test_audit_plan_code_consistency.py tests/test_local_pr_review.py tests/test_pre_push_audit_workflow.py -q - 45 passed.
- python -m py_compile scripts/audit_plan_code_consistency.py tests/test_audit_plan_code_consistency.py - passed.
- python scripts/audit_plan_code_consistency.py --base-ref origin/main plans/PR-Plan-Code-Consistency-Deleted-Paths.md - passed.
- python scripts/maturity_sweep.py scripts --tests-root tests --baseline tests/maturity_sweep/baseline_scripts.json --min-score 8 --sensitive-glob 'scripts/**' - passed; ratchet gate reported no new brittleness above baseline.

## Estimated diff size

| File | LOC |
|---|---:|
| `.github/workflows/pre_push_audit.yml` | 4 |
| `plans/PR-Plan-Code-Consistency-Deleted-Paths.md` | 162 |
| `scripts/audit_plan_code_consistency.py` | 192 |
| `scripts/local_pr_review.sh` | 4 |
| `tests/test_audit_plan_code_consistency.py` | 154 |
| `tests/test_local_pr_review.py` | 23 |
| **Total** | **539** |

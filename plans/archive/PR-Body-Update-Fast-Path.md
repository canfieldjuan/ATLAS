# PR-Body-Update-Fast-Path

## Why this slice exists

PR #2329 exposed a workflow failure in the body-only fix loop. A manual
`gh pr edit --body-file` update removed the hidden `open_pr.sh` wrapper marker,
so the trusted `pr-body-contract` check failed even though the visible body
looked valid. The compliant repair path, `scripts/open_pr.sh`, then reran the
full local review and unit-gate mirror for a body-only reconciliation edit,
which made the safe path expensive enough to discourage use.

This workflow/process slice makes the body-only path cheap and still bounded:
publish an existing PR body through a dedicated helper that stamps the wrapper
marker, runs the body/reconciliation checks that can actually be affected by a
body edit, verifies PR ownership/head identity, and leaves code/full-review
checks to the push/open path.

Diff-budget override: this workflow helper is indivisible in one PR because the
script, AGENTS contract hook, impacted-test enrollment, and fake GitHub wrapper
fixtures must ship together for the new path to be usable and reviewable.

### Problem-derived contract

- Root cause: Atlas has one approved PR-body mutation wrapper, but it is shaped
  for PR create/update after code changes. For an existing PR body-only
  reconciliation edit, the cheap path is raw `gh pr edit` and the safe path is a
  full local-review replay. That incentive mismatch caused the missing wrapper
  marker failure on #2329.
- Correct fix must touch/change: add a focused existing-PR body update helper
  that reuses the canonical body marker and body/reconciliation audits; document
  when to use it; add wrapper tests proving it stamps full bodies, refuses
  target-changing edits, verifies ownership/head identity, runs focused checks,
  and does not call `local_pr_review.sh`.
- Must not change: do not change PR body schema, live reconciliation semantics,
  required GitHub checks, PR title/base/label editing, product behavior, or the
  `open_pr.sh` create/update path for code-bearing pushes.

## Scope (this PR)

Ownership lane: workflow/pr-body-update-fast-path
Slice phase: Workflow/process

1. Add a dedicated helper for existing PR body-only updates.
2. Document the helper as the required path when only the PR body changes after
   a PR is already open.
3. Add focused wrapper tests and impacted-test enrollment for the new helper.

### Review Contract

- Acceptance criteria:
  1. `scripts/update_pr_body.sh` appends the same
     `<!-- atlas-open-pr-wrapper: v1 -->` marker used by `scripts/open_pr.sh`
     when a full human PR body lacks it, proved by
     `tests/test_update_pr_body_wrapper.py::test_update_wrapper_stamps_full_body_before_publish`.
  2. The helper runs focused body checks only: branch-name audit, PR-body audit,
     AI reconciliation audit, fix-loop disposition preflight, optional live
     reconciliation, and session ownership/head checks; it does not invoke
     `scripts/local_pr_review.sh`, proved by
     `tests/test_update_pr_body_wrapper.py::test_update_wrapper_does_not_run_full_local_review`.
  3. The helper refuses create/target-changing/title/base/body argv edits and
     only updates the existing PR for the current branch, proved by
     `tests/test_update_pr_body_wrapper.py::test_update_wrapper_rejects_target_changing_args`.
  4. The helper fails before mutation when the PR head reported by GitHub does
     not match the reviewed local head, proved by
     `tests/test_update_pr_body_wrapper.py::test_update_wrapper_rejects_head_drift_before_edit`.
  5. `AGENTS.md` directs body-only PR updates through the helper while leaving
     `open_pr.sh` as the create/body-update path after code changes.
- Reachability proof: run the wrapper tests against a fake git/gh harness and
  inspect the dry-run output showing the final mutation would be
  `gh pr edit <number> --body-file -`.
- Affected surfaces: `AGENTS.md`, `scripts/update_pr_body.sh`,
  `scripts/select_impacted_tests.py`, and wrapper tests.
- Risk areas: wrapper marker parity, ownership/head drift, accidental target
  mutation, accidentally rerunning full local review, and live reconciliation
  mismatch.
- Reviewer rules triggered: R1, R2, R10, R12, R14.

### Boundary-change enumeration

Required when this diff changes a guard, validator, normalizer, resolver,
router/classifier, or admission boundary. Name each changed boundary path or
seam in the enumeration; otherwise write "N/A - no boundary change."

- Boundary path/seam: existing PR body mutation command.
- Replaced-path behaviors: raw `gh pr edit --body-file` is replaced for
  body-only updates with a wrapper that stamps and audits the body before
  mutation.
- Guard-relevant fields: current branch, PR number, PR head SHA, PR base,
  PR body marker, PR author, session state file, and body file hash.
- Caller x input shape: shell invocation
  `bash scripts/update_pr_body.sh <body-file>` on a branch with exactly one
  open PR targeting `main`.

### Deployed-config probing

Required for guard, validator, resolver, admission-boundary, or env/config
fallback changes; otherwise write "N/A - no guard/config boundary change."

- Deployed/default config values: N/A - local GitHub CLI wrapper, no deployed
  configuration or env fallback.
- Explicit value probe: wrapper tests provide explicit PR/body/head fixtures.
- Absent value probe: wrapper tests cover missing/invalid target-changing args
  and head drift before mutation.
- Default-session/default-context probe: session ownership guard is invoked
  before mutation when PR metadata is known.
- Side-effect ordering: body audit, ownership guard, and head verification all
  run before the `gh pr edit` mutation.

### Files touched

- `AGENTS.md`
- `plans/PR-Body-Update-Fast-Path.md`
- `scripts/select_impacted_tests.py`
- `scripts/update_pr_body.sh`
- `tests/test_update_pr_body_wrapper.py`

## Mechanism

Add `scripts/update_pr_body.sh` for existing PR body-only updates. It prepares a
temporary publish body with the same wrapper marker as `open_pr.sh`, validates
the current branch name and PR body shape, runs AI/fix-loop reconciliation
checks, discovers the existing current-branch PR, verifies ownership and head
identity, optionally runs live reconciliation when `gh` can read the PR, and
then publishes via `gh pr edit <number> --body-file -`.

The helper is intentionally narrower than `open_pr.sh`: it never creates PRs,
never changes title/base/labels/reviewers, and never runs
`scripts/local_pr_review.sh`. Code changes still use the existing
`push_pr.sh` -> `open_pr.sh` path.

## Intentional

- Keep this as a new narrow helper instead of weakening `open_pr.sh`; the create
  path still needs full local review.
- Keep live reconciliation semantics unchanged; this helper only calls the
  existing checker before publishing body text.
- Do not add automatic review-thread resolution; resolving remains an explicit
  post-fix action after the body and code truth match.

## Deferred

- A generator that builds exact AI reconciliation root ledgers from live Codex
  thread history.
- A broader local-review scheduler that avoids duplicate full unit mirrors
  across concurrent worktrees.

Parked hardening: none.

## Verification

- `python -m pytest tests/test_update_pr_body_wrapper.py -q` - passed, 6 tests.
- `python -m pytest tests/test_select_impacted_tests.py -q` - passed, 64 tests.
- `bash -n scripts/update_pr_body.sh` - passed.
- Pending before push: plan/body audits and local review wrapper.

## Estimated diff size

| File | LOC |
|---|---:|
| `AGENTS.md` | 11 |
| `plans/PR-Body-Update-Fast-Path.md` | 165 |
| `scripts/select_impacted_tests.py` | 3 |
| `scripts/update_pr_body.sh` | 235 |
| `tests/test_update_pr_body_wrapper.py` | 283 |
| **Total** | **697** |

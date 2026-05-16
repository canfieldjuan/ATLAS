# PR: Local PR Review Dirty Worktree Guard

## Why this slice exists

`scripts/local_pr_review.sh` compares the committed branch diff against
`origin/main`. When it runs with uncommitted edits, some plan/code checks can
skip or report against an incomplete committed diff. The script should fail
early unless the caller explicitly opts into dirty-worktree mode.

## Scope

1. Add a clean-worktree guard to the local PR review script.
2. Add an explicit `--allow-dirty` escape hatch.
3. Preserve the existing optional base-ref argument.
4. Add focused script tests for dirty failure and allow-dirty behavior.

## Mechanism

The script parses `--allow-dirty` and a single optional base ref before running
checks. When dirty mode is not allowed, it checks `git status --porcelain` and
exits with guidance if any tracked, staged, or untracked changes are present.

## Intentional

- No changes to the pre-push audit wrapper.
- No automatic pytest or npm expansion.
- No changes to the installed git hook.

## Deferred

- Changed-file test suggestions.
- PR-size warnings.
- A broader shell hygiene auditor.

## Verification

- Run the focused local PR review tests.
- Run a shell syntax check for the script.
- Run the local PR review script with the explicit dirty override because this
  PR intentionally edits files before commit.
- Run diff whitespace checks.

### Files Touched

- `plans/PR-Audit-Local-Review-Dirty-Guard.md`
- `scripts/local_pr_review.sh`
- `tests/test_local_pr_review.py`

## Estimated Diff Size

| Area | Estimated LOC |
|---|---:|
| Script | ~35 |
| Tests | ~110 |
| Plan and coordination | ~55 |
| **Total** | ~200 |

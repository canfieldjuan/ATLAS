# PR-Stale-Base-Push-Guard

## Why this slice exists

#1474 exposed the recurring stale-base failure mode at its worst: the branch
local review used a stale `origin/main`, so the drift audit saw zero base
changes locally while GitHub review showed the PR would have reverted three
merged PRs. The review was caught before merge, but the workflow still relies on
the builder remembering to fetch/rebase before every push.

This slice moves that reminder into the push wrapper. Before `push_pr.sh` hands
control to the local-review hook or pushes a branch, it refreshes the default
remote base so the existing drift audit compares against the real current
`origin/main`.

## Scope (this PR)

Ownership lane: workflow/process
Slice phase: Workflow/process

1. Update `scripts/push_pr.sh` to refresh `origin/main` before the wrapper or
   managed pre-push hook runs local PR review.
2. Keep dry-run mode side-effect free while making the planned fetch visible in
   dry-run output.
3. Add tests proving the fetch happens before review/push and a failed fetch
   aborts before any push.
4. Enroll `tests/test_push_pr_wrapper.py` in the pre-push audit workflow's
   explicit PR-review tooling pytest list.

### Review Contract

- Acceptance criteria:
  - [ ] `push_pr.sh` refreshes `origin/main` before local PR review runs in both
        wrapper-review and managed-hook paths.
  - [ ] Dry-run mode does not fetch but prints the planned base refresh.
  - [ ] A fetch failure exits non-zero before `git push`.
  - [ ] Existing no-`--no-verify`, missing-body, and single-review behavior is
        preserved.
  - [ ] The push-wrapper test file runs in CI through
        `.github/workflows/pre_push_audit.yml`.
- Affected surfaces: PR push wrapper, pre-push audit workflow test list, and
  their tests.
- Risk areas: accidentally duplicating local review again, doing network work in
  dry-run, bypassing hooks, or hiding fetch failures.
- Reviewer rules triggered: R2, R10, R14.

### Files touched

- `.github/workflows/pre_push_audit.yml`
- `plans/PR-Stale-Base-Push-Guard.md`
- `scripts/push_pr.sh`
- `tests/test_pre_push_audit_workflow.py`
- `tests/test_push_pr_wrapper.py`

## Mechanism

`push_pr.sh` already centralizes Atlas PR pushes and chooses between:

- wrapper-run `local_pr_review.sh`, when no managed hook will run it; and
- hook-run `local_pr_review.sh`, when the managed pre-push hook is installed.

This PR adds a small base-refresh step before either path:

```bash
git fetch --quiet origin main
```

The fetch updates the local `origin/main` ref before the drift audit computes
`git merge-base HEAD origin/main`. If another PR landed since the builder last
fetched, the existing `audit_pr_session_drift.py` check can now see the base
changes and block stale diffs before the branch is pushed.

Dry-run mode only prints the planned fetch and exits without network I/O. A real
fetch failure is fatal; pushing with stale base information is exactly the
failure this slice exists to prevent.

The existing pre-push audit workflow runs a hand-maintained pytest file list for
PR-review tooling. This PR adds `tests/test_push_pr_wrapper.py` to that list and
adds a workflow test asserting the file stays enrolled.

## Intentional

- This does not auto-rebase. Rebasing can create conflicts and rewrite history;
  the safe automated behavior is to refresh the base and let the existing drift
  audit fail with a concrete reason.
- This lives in `push_pr.sh`, not `local_pr_review.sh`, so ad hoc local reviews
  stay offline-capable. The pre-push path is the place where stale remote state
  can turn into a dangerous pushed PR.
- The default is hard-coded to `origin/main` because Atlas PRs target main and
  the existing local review bundle defaults to that same base.

## Deferred

- A stronger future guard could make `local_pr_review.sh` itself optionally
  fetch when invoked directly, but that would change offline/advisory review
  behavior. This slice closes the actual push/open failure path.

Parked hardening: none.

## Verification

- `pytest tests/test_push_pr_wrapper.py` passed, 9 tests.
- `python -m pytest tests/test_push_pr_wrapper.py tests/test_local_pr_review.py -q` passed, 18 tests.
- `python -m pytest tests/test_local_pr_review.py tests/test_audit_ai_reconciliation.py tests/test_audit_review_rules_triggered.py tests/test_summarize_review_misses.py tests/test_check_ai_reconciliation_live.py tests/test_pre_push_audit_workflow.py tests/test_check_review_body_r14.py tests/test_push_pr_wrapper.py -q` passed, 92 tests.
- `python scripts/sync_pr_plan.py plans/PR-Stale-Base-Push-Guard.md origin/main --check` passed.
- `python scripts/audit_plan_code_consistency.py plans/PR-Stale-Base-Push-Guard.md` passed.
- `python scripts/audit_review_rules_triggered.py --plan plans/PR-Stale-Base-Push-Guard.md` passed.
- `git diff --check origin/main...HEAD` passed.
- `bash scripts/push_pr.sh /tmp/atlas-pr-stale-base-push-guard.md -u origin HEAD` passed; the wrapper printed `Refreshing origin/main before local PR review...`, then the managed pre-push hook ran `local_pr_review.sh` once and passed with `base changed files since branch point: 0`.

## Estimated diff size

| File | LOC |
|---|---:|
| `.github/workflows/pre_push_audit.yml` | 2 |
| `plans/PR-Stale-Base-Push-Guard.md` | 120 |
| `scripts/push_pr.sh` | 11 |
| `tests/test_pre_push_audit_workflow.py` | 6 |
| `tests/test_push_pr_wrapper.py` | 151 |
| **Total** | **290** |

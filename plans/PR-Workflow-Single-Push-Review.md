# PR-Workflow-Single-Push-Review

## Why this slice exists

The Gate A email-campaign PR push path repeated the same mechanical local
review bundle multiple times: once manually, once through `push_pr.sh`, and
again through the installed pre-push hook. The operator called out that this
will recur after compaction unless the rule is durable.

This workflow slice removes the redundant wrapper-side local-review run when
the managed hook is already installed, while preserving a fail-safe fallback
when the hook is absent or explicitly skipped.

## Scope (this PR)

Ownership lane: dev-workflow/pr-review-efficiency
Slice phase: Workflow/process

1. Update `scripts/push_pr.sh` so a managed Atlas pre-push hook is the single
   local-review runner during a push.
2. Keep wrapper-side local review as the fallback when the managed hook is
   absent or `ATLAS_SKIP_LOCAL_PR_REVIEW=1` would bypass it.
3. Lock the behavior in focused wrapper tests and document the compact-safe
   rule in the builder contract/bootstrap.

### Review Contract

- Acceptance criteria:
  - [ ] With a managed Atlas pre-push hook present and not skipped,
        `push_pr.sh` does not run `local_pr_review.sh` before `git push`; the
        hook is the only runner.
  - [ ] With no managed hook, `push_pr.sh` still runs `local_pr_review.sh`
        before pushing, so local review is not silently lost.
  - [ ] With `ATLAS_SKIP_LOCAL_PR_REVIEW=1`, `push_pr.sh` runs local review
        before pushing because the hook will intentionally skip.
  - [ ] `push_pr.sh` rejects forwarded `--no-verify`, because relying on the
        hook as the single runner cannot allow hook bypass.
  - [ ] Fresh-session docs tell builders to use `push_pr.sh` and avoid a
        separate manual local-review run immediately before the wrapper.
- Affected surfaces: developer workflow scripts, local pre-push behavior,
  builder bootstrap/contract docs.
- Risk areas: CI/process enforcement, compaction discipline, failed-push loops.
- Reviewer rules triggered: R1, R2, R10, R12.

### Files touched

- `AGENTS.md`
- `docs/SESSION_BOOTSTRAP.md`
- `plans/PR-Workflow-Single-Push-Review.md`
- `scripts/push_pr.sh`
- `tests/test_push_pr_wrapper.py`

## Mechanism

`push_pr.sh` checks `.git/hooks/pre-push` via `git rev-parse --git-path` and
the managed `ATLAS_LOCAL_PR_REVIEW_HOOK` marker installed by
`scripts/install_local_pr_hook.sh`.

If that managed hook is present and `ATLAS_SKIP_LOCAL_PR_REVIEW` is not set,
the wrapper exports `ATLAS_CURRENT_PR_BODY_FILE` and goes straight to
`git push`; the hook runs `local_pr_review.sh` once with that body context.

If the hook is missing, unmanaged, or would skip, the wrapper runs
`local_pr_review.sh --current-pr-body-file <body>` itself before pushing. That
keeps the safety check while removing the duplicate work in the normal
installed-hook path.

The wrapper rejects forwarded `--no-verify` because that would bypass the
managed hook after this slice makes the hook the normal single runner.

## Intentional

- The installed pre-push hook remains the source of enforcement for this
  checkout. Removing the wrapper preflight unconditionally would skip local
  review on checkouts without the managed hook, so this slice keeps the
  fallback.
- This PR does not change the full extracted gauntlet policy. It only prevents
  duplicate mechanical local-review execution around `git push`.

## Deferred

None.

Parked hardening: none.

## Verification

- `python -m pytest tests/test_push_pr_wrapper.py tests/test_install_local_pr_hook.py -q` -- 10 passed.
- `ATLAS_PUSH_PR_DRY_RUN=1 bash scripts/push_pr.sh plans/PR-Workflow-Single-Push-Review.md -u origin HEAD` -- printed the managed-hook single-run path without a wrapper-side `local_pr_review.sh` command.
- `bash scripts/push_pr.sh tmp/workflow-single-push-review-pr-body.md -u origin claude/pr-workflow-single-push-review` -- runs the managed pre-push hook once before pushing.

## Estimated diff size

| File | LOC |
|---|---:|
| `AGENTS.md` | 17 |
| `docs/SESSION_BOOTSTRAP.md` | 17 |
| `plans/PR-Workflow-Single-Push-Review.md` | 101 |
| `scripts/push_pr.sh` | 42 |
| `tests/test_push_pr_wrapper.py` | 98 |
| **Total** | **275** |

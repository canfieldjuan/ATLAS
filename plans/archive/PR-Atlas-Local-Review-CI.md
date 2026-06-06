# PR-Atlas-Local-Review-CI

## Why this slice exists

AGENTS.md says GitHub Actions should run the same local review wrapper
after a PR opens, but the current workflow still runs only
`scripts/pre_push_audit.sh`. That means the CI gate misses checks that
live only in `scripts/local_pr_review.sh`, including the just-landed
current-PR `Slice phase` body validation.

This slice makes CI enforce the same mechanical bundle builders run
locally.

## Scope (this PR)

Ownership lane: atlas-workflow

Slice phase: Workflow/process.

1. Run `scripts/local_pr_review.sh` from the pre-push audit workflow
   instead of the narrower pre-push wrapper.
2. Provide `GH_TOKEN` in that workflow so the drift audit can read open
   PR metadata through `gh`.
3. Make current-PR detection work in detached GitHub Actions checkouts by
   honoring `GITHUB_HEAD_REF`.
4. Add a fixture proving `GITHUB_HEAD_REF` identifies the current PR even
   when `git branch --show-current` is empty.

### Files touched

- `.github/workflows/pre_push_audit.yml`
- `scripts/audit_pr_session_drift.py`
- `tests/test_audit_pr_session_drift.py`
- `plans/PR-Atlas-Local-Review-CI.md`

## Mechanism

The workflow keeps its name and trigger surface, but its audit step calls
the `scripts/local_pr_review.sh` wrapper. That wrapper already runs
`scripts/pre_push_audit.sh`, then adds drift, cross-layer caller,
plan/code consistency, and whitespace checks.

`audit_pr_session_drift.py` currently identifies the current PR by local
branch name or matching head SHA. In GitHub Actions pull-request checkouts
the repository can be detached, so the local branch name may be empty and
the checked-out SHA may be a synthetic merge commit. Reading
`GITHUB_HEAD_REF` first gives the audit the PR head branch that GitHub
already exposes.

## Intentional

- No workflow rename in this slice. Keeping the existing "Pre-push Audit"
  workflow avoids branch-protection churn while strengthening the command it
  runs.
- No new audit script. `local_pr_review.sh` remains the single mechanical PR
  review wrapper.
- The workflow grants only `contents: read` and `pull-requests: read` so
  `gh` metadata reads do not depend on public-repo defaults; no write
  permissions are added.
- Cross-session drift overlap remains blocking in CI. A PR may need to rebase
  when main or another open same-lane PR moves, but this is the intended
  collision signal for the workflow arc.

## Deferred

- Future PR: consider renaming the workflow/job from "Pre-push Audit" to
  "Local PR Review" if branch protection and reviewer expectations allow it.
- Parked hardening: none.

## Verification

- `python -m pytest tests/test_audit_pr_session_drift.py -q` - 23 passed.
- `bash scripts/local_pr_review.sh --allow-dirty` - passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Workflow | ~10 |
| Drift audit | ~8 |
| Tests | ~45 |
| Plan doc | ~75 |
| **Total** | ~138 |

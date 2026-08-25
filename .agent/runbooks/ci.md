# CI and pull-request inspection

Atlas uses GitHub Actions. The current GitHub CLI session is authenticated and
can inspect the repository; inspection does not grant permission to rerun,
dispatch, edit, push, review, or merge.

## Fast inspection

```bash
./ops ci status
./ops ci status 25
./ops ci run <run-id>
./ops ci run <run-id> --log-failed
```

For a PR, first establish exact ownership and head:

```bash
gh pr view <number> --repo canfieldjuan/ATLAS \
  --json number,url,headRefName,headRefOid,baseRefName,state,isDraft
gh pr checks <number> --repo canfieldjuan/ATLAS
```

Do not infer current status from a prior message, commit, or run. Match the PR
head OID to the run head SHA and re-poll review/reconciliation immediately
before a verdict or merge decision.

## What is required

`ci/gates.yml` is the machine-readable enforcement registry. At verification
time its branch-required contexts include live reconciliation, secret scanning,
baseline-growth protection, diff budget, plan admission, session lane, review
contract, PR body contract, and Unit Gate. Read the registry live rather than
copying this sentence into a decision; the set can change.

Path-specific product/package workflows supplement those meta gates. A workflow
being active does not mean it triggers on every diff. The Unit Gate runs on
every PR and chooses impacted tests or the full non-integration/e2e suite;
Repo-Wide Unit Backstop runs the full unit suite on schedule/on demand.

The builder's local gate is `scripts/local_pr_review.sh`, normally invoked once
by `scripts/push_pr.sh`/the managed pre-push hook. GitHub re-runs the trusted-base
audit. Do not delete that duplication or run the same local bundle twice
immediately before one push.

## Actions that mutate CI or PR state

The following are not discovery commands and are intentionally absent from
`./ops`: workflow dispatch, run rerun/cancel/delete, PR comment/review/edit,
push, merge, and branch protection changes. Follow `AGENTS.md`, the session
state ownership guard, and the active PR contract before any of them.

## Failure routing

- Red required check: inspect the exact run and failed step; do not merge.
- `steps: []` or failure before runner allocation: classify it as hosted
  infrastructure/runner failure and compare local workflow-equivalent gates.
- Check absent: inspect workflow path filters and `ci/gates.yml`; absence is not
  green.
- Advisory workflow red: report it separately from branch-required status, but
  still assess whether it exposes a real security/data/deployment defect.
- GitHub authentication fails: use `gh auth status`; do not re-login or replace
  tokens without operator direction.
- PR ownership guard fails: stop and ask the operator. Lane similarity is not
  ownership.

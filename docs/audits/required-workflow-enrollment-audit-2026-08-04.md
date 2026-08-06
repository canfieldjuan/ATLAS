# Required Workflow Enrollment Audit - 2026-08-04

## 2026-08-05 Recheck

#2290 closed the `unit-gate` selector blocker by mapping CI/security governance
docs to their owning tests and by making the watcher governance doc owner run
against the checked-out repository docs. The recheck decision is:

- `unit-gate`: promote to `branch_required` and require it in live branch
  protection.
- `pre-push-audit`: keep at `ci_blocking_not_required` until the trusted-base
  PR-side docs/test consistency blocker has a safe data-only probe.

Fresh evidence on 2026-08-05:

- #2290's `unit-gate` run passed in about 2m21s.
- The latest sampled `unit_gate.yml` runs show the selector now produces bounded
  runs for governance-doc changes. Recent failures were legitimate PR test
  failures, not runner flakes.
- A fresh live branch-protection payload now includes `unit-gate` pinned to the
  GitHub Actions app source with `strict: false`, and
  `scripts/check_required_status_checks.py` passes against that payload.
- `pre-push-audit` remains useful and green in recent samples, but its open
  blocker is different: because PR events run trusted base code, PR-side changes
  to gate docs/tests can still be observed only after merge unless a safe
  data-only consistency probe is added.

## 2026-08-04 Initial Decision

Initial decision on 2026-08-04: keep `pre-push-audit` and `unit-gate` at
`ci_blocking_not_required` for that slice. Do not add either context to
`ci/gates.yml` `branch_required` or live branch protection in that slice.

As of the 2026-08-05 recheck above, this initial decision remains current for
`pre-push-audit` only. It is superseded for `unit-gate` because #2290 closed the
specific selector blocker named by this audit.

This is not a downgrade. Both checks remain real CI signals, and red runs must
be fixed before calling a PR ready. The decision is that neither is ready to be
made part of the hard branch-protection contract without first closing the
specific enrollment blockers below.

## Evidence

Registry state at the 2026-08-04 initial decision:

```bash
git show 9b4a37e6f7e3f7e34c1e87910591c65fcb8fa5b5:ci/gates.yml | sed -n '72,92p'
```

- `pre-push-audit`: `ci_blocking_not_required`
- `unit-gate`: `ci_blocking_not_required`

Live branch protection as of the 2026-08-04 initial decision contained every existing
`branch_required` registry context pinned to the GitHub Actions app source, and
does not include `pre-push-audit` or `unit-gate`:

```bash
gh api repos/canfieldjuan/ATLAS/branches/main/protection/required_status_checks
python scripts/check_required_status_checks.py \
  --payload-file /tmp/atlas-required-status-checks-live-required-workflow.json
```

The checker passed for the then-current registry-required set. The 2026-08-05
recheck above is the current deployed state for `unit-gate`.

Recent `pre-push-audit` sample:

```bash
gh run list --workflow pre_push_audit.yml --limit 20 \
  --json databaseId,status,conclusion,createdAt,updatedAt,event,headBranch,displayTitle
```

The first ten terminal sampled runs
(`30961794217`, `30961778404`, `30961650737`, `30961567962`, `30961537815`,
`30960553412`, `30960353619`, `30960181583`, `30960167467`, `30959465034`) had
3 success and 7 failure outcomes. The sampled median duration was about 106
seconds, with a max of 146 seconds. The recent failures were legitimate
process/test failures, not runner flakes; one example failed
`tests/test_security_guardrails_workflow.py` after the required status docs
changed.

Recent `unit-gate` sample:

```bash
gh run list --workflow unit_gate.yml --limit 20 \
  --json databaseId,status,conclusion,createdAt,updatedAt,event,headBranch,displayTitle
```

The first ten terminal sampled runs
(`30961651974`, `30961539028`, `30960553450`, `30960168490`, `30959465237`,
`30958679496`, `30958429073`, `30956838206`, `30956772059`, `30955811357`) had
7 success, 1 failure, and 2 cancelled outcomes. The sampled median duration was
about 631 seconds, with a max of 699 seconds. The recent failure was the same
stale security-guardrails docs test that `pre-push-audit` reported.

## Trust model

`pre-push-audit` runs on `pull_request_target` from trusted base code. It checks
out the PR base SHA, materializes the PR head as data, runs
`scripts/local_pr_review.sh` against that PR worktree, and then runs trusted-base
PR-review tooling tests. That protects the gate from PR-side edits to its own
checker code, but it also means a PR can change gate docs/tests in ways the
trusted-base test list only observes after merge.

`unit-gate` runs on `pull_request` against the PR head SHA. It installs the PR's
dependencies, selects impacted tests when possible, falls back to the full
non-integration/e2e unit suite when selection escalates to `FULL`, and runs the
baseline growth guard when no changed file maps to a reachable test. It cancels
superseded in-progress runs on the same PR ref.

## Enrollment blockers

`pre-push-audit` is trusted-base and valuable, but requiring it would not have
prevented the stale docs/test mismatch that followed the required-status record
merge. On pull requests, the workflow executes trusted base scripts and tooling
tests. That is the right security model for PR-authored gate changes, but it
means a PR that changes docs/tests for the gate can pass before merge and fail
the push-to-main variant after merge. That behavior should be fixed or made
explicit before treating the context as a hard merge contract.

At the 2026-08-04 initial decision, `unit-gate` was not ready for branch
protection because its selector did not run the security-guardrails docs test
for the docs-only required-status PR. The PR run selected no reachable tests and
passed growth-only, while a later full-suite path failed on
`tests/test_security_guardrails_workflow.py`. Making `unit-gate` required before
fixing that selector coverage would have added wait time without closing the
observed gap. #2290 later closed this specific blocker; see the 2026-08-05
recheck above for the current `unit-gate` decision.

## Result

The 2026-08-04 initial result was not "require both checks." It was to close the
observed coverage gap first:

- Map `docs/SECURITY_GUARDRAILS.md` and branch-protection docs to
  `tests/test_security_guardrails_workflow.py` in the unit-gate selector
  (closed by #2290 before the 2026-08-05 recheck).
- Decide whether trusted-base `pre-push-audit` needs a PR-side docs/test
  consistency probe that still avoids executing untrusted PR code.
- Re-run this enrollment decision after those blockers are closed.

# PR-Security-Gitleaks-Required-Checks

## Why this slice exists

#1656's credential-leak prevention notes identify a merge-enforcement gap:
Gitleaks runs on PRs, but live branch protection required only
`live-reconciliation`, so a red Gitleaks PR scan or baseline-growth guard could
be ignored or overridden during merge. After #1825 added the local pre-commit
guard, the remaining prevention layer is the merge-time guard.

Root cause: required status checks are mutable GitHub repository settings, not
repo-owned code, and Atlas had no drift detector proving that the security
scanner contexts stay required. The live setting was updated before this PR so
`main` now requires `live-reconciliation`, `Gitleaks PR secret scan`, and
`Gitleaks baseline growth guard`; this slice makes that permanent by adding a
scheduled/manual audit that fails if branch protection drops either Gitleaks
context.

Review update root cause: the first checker draft only proved bare context
names and the baseline guard lived in a mixed `pull_request` /
`pull_request_target` workflow, so GitHub could report a skipped
`pull_request` job with the same required context while branch protection could
also be recreated with legacy or wrong-app check sources. This update fixes the
root by splitting the baseline guard into a `pull_request_target`-only workflow
and requiring the live branch-protection payload to pin required checks to the
GitHub Actions app source.

Diff budget note: the final diff exceeds the 400 LOC target because the review
fix is not safely separable from its regression fixtures: the baseline guard
must move out of the mixed-event workflow, the required-check checker must
validate source pins, the workflow posture auditor must recognize the new
canonical pull_request_target guard file, and the same PR must prove those
detector failure modes.

## Scope (this PR)

Ownership lane: security/gitleaks-required-checks
Slice phase: Workflow/process

1. Add a small branch-protection required-status checker for the three required
   contexts: `live-reconciliation`, `Gitleaks PR secret scan`, and
   `Gitleaks baseline growth guard`, including their GitHub Actions app source.
2. Add a scheduled/manual workflow that reads the live `main` branch protection
   required-status payload and runs the checker when the repository has a
   branch-protection read token configured and the workflow is running from
   `main` or a trusted schedule/main push.
3. Split the baseline-growth guard into a `pull_request_target`-only workflow
   so its required context cannot be satisfied by a skipped `pull_request` job.
4. Extend enrolled security workflow tests so missing Gitleaks contexts,
   legacy-only contexts, wrong-app contexts, skipped baseline contexts, and the
   workflow wiring fail in fixtures.
5. Update the workflow posture auditor allowlist so the new dedicated baseline
   workflow is the only approved pull_request_target trusted-base guard shape.
6. Document the required-check posture in `docs/SECURITY_GUARDRAILS.md`.
7. Archive the merged #1825 Gitleaks pre-commit plan doc as teardown
   housekeeping.

### Review Contract

- Acceptance criteria:
  - [ ] The live `main` branch protection setting includes
        `live-reconciliation`, `Gitleaks PR secret scan`, and
        `Gitleaks baseline growth guard`.
  - [ ] The checker fails when either Gitleaks required-check context is
        missing from the required-status payload.
  - [ ] The checker fails when required contexts are present only in the legacy
        `contexts` list or when `checks[].app_id` is missing, unpinned, or not
        the GitHub Actions app.
  - [ ] The baseline growth guard required context is emitted only by a
        `pull_request_target`-only workflow, not by a skipped job in a
        mixed-event PR workflow.
  - [ ] The workflow posture auditor accepts the split baseline workflow as the
        approved trusted-base pull_request_target shape and rejects the old
        mixed-workflow location.
  - [ ] The workflow fetches the live branch-protection required-status payload
        and runs the checker on a schedule and by manual dispatch when
        `ATLAS_BRANCH_PROTECTION_READ_TOKEN` is configured.
  - [ ] Non-main manual dispatch cannot reach checkout or the
        Administration-read token-backed audit steps.
  - [ ] The added tests are CI-enrolled through the existing pre-push audit
        workflow.
- Affected surfaces: repository branch-protection drift detection, security
  guardrail docs, pre-push audit tests, and plan archive housekeeping.
- Risk areas: treating a settings-only change as durable without a repo-owned
  detector, requiring the wrong GitHub check context, or making CI depend on
  branch-protection write permissions.
- Reviewer rules triggered: R1, R2, R3, R10, R11, R14.

### Files touched

- `.github/workflows/branch_protection_required_checks.yml`
- `.github/workflows/gitleaks_baseline_growth_guard.yml`
- `.github/workflows/security_guardrails.yml`
- `docs/SECURITY_GUARDRAILS.md`
- `plans/INDEX.md`
- `plans/PR-Security-Gitleaks-Required-Checks.md`
- `plans/archive/PR-Security-Gitleaks-Precommit.md`
- `scripts/audit_workflow_security_posture.py`
- `scripts/check_required_status_checks.py`
- `tests/test_audit_workflow_security_posture.py`
- `tests/test_security_guardrails_workflow.py`

## Mechanism

`scripts/check_required_status_checks.py` reads the JSON returned by
`GET /repos/{owner}/{repo}/branches/main/protection/required_status_checks`.
It parses required status names from both the legacy `contexts` list and the
newer `checks[].context` list for diagnostics, then exits non-zero unless every
required context is present in `checks[]` with the GitHub Actions app ID
(`15368`). That fails closed when branch protection is recreated as legacy bare
contexts, with `app_id: -1`, or with another app as the required-check source.

The baseline growth guard now lives in
`.github/workflows/gitleaks_baseline_growth_guard.yml`, whose only trigger is
`pull_request_target`; the job also carries an explicit
`github.event_name == 'pull_request_target'` guard so the existing workflow
posture auditor can prove the approved shape. The existing security sweep
workflow keeps the PR secret scan and trusted-main/advisory scanners but no
longer defines a
`Gitleaks baseline growth guard` job, so the merge-required baseline context
cannot be satisfied by a skipped `pull_request` job.

`scripts/audit_workflow_security_posture.py` now allowlists
`gitleaks_baseline_growth_guard.yml:gitleaks-baseline-guard` as the single
approved pull_request_target baseline guard. Its fixtures reject the old
`.github/workflows/security_guardrails.yml` location and still reject extra
pull_request_target jobs without the trusted-base checkout guard.

The branch-protection audit workflow runs on `workflow_dispatch`, weekly
schedule, and pushes to `main` touching the checker/workflow/security-doc
files. GitHub's branch protection endpoint requires Administration read
permission, so the workflow uses `ATLAS_BRANCH_PROTECTION_READ_TOKEN` when it
is configured and otherwise skips with a notice instead of making `main` red.
With the token configured, it uses the same manual-dispatch ref guard as other
secret-bearing workflows: `github.event_name != 'workflow_dispatch' ||
github.ref == 'refs/heads/main'`. Checkout is pinned to
`github.event.repository.default_branch`, so a manual dispatch on an arbitrary
branch cannot run branch-controlled code with the Administration-read token. On
a trusted ref, it fetches the live required-status-check payload with `gh api`,
then runs the checker against that payload. The workflow detects drift but does
not modify branch protection.

## Intentional

- The branch-protection mutation itself is a GitHub repository setting, not a
  PR diff. This PR owns the durable drift detector and documentation.
- The branch-protection audit workflow is schedule/manual/main-push only. It
  does not run on arbitrary PRs because PRs cannot change the live
  branch-protection setting and the focused fixture tests already prove the
  parser/detector behavior.
- The audit is read-only. It reports drift instead of trying to auto-repair
  branch protection from CI.
- The workflow skips when `ATLAS_BRANCH_PROTECTION_READ_TOKEN` is absent because
  the default Actions token cannot be granted the Administration read permission
  required by GitHub's branch-protection endpoint.
- This deliberately mirrors the #1809/#1820/#1823 ref-guard pattern because the
  branch-protection read token is more sensitive than the default Actions token.
- The live required-check source is pinned to GitHub Actions app ID `15368`,
  matching the current branch-protection API payload for all three required
  contexts.
- The pull_request_target allowlist moved with the baseline guard instead of
  broadening to any job in any workflow.

## Deferred

- #1656 follow-up: rotate/revoke the credentials exposed in historical commit
  `d63a9b77b9727766e14e523626c22dd6c1c80da8`.
- #1656 follow-up: ratchet the remaining advisory scanners after their
  backlogs are burned down.

Parked hardening: none.

## Verification

- Focused workflow/security tests: `29 passed in 0.10s`.
- Python compile check for the checker, workflow posture auditor, and touched
  tests: passed.
- Live branch-protection required-check audit: passed against
  `repos/canfieldjuan/ATLAS/branches/main/protection/required_status_checks`.
- Workflow security posture audit: passed locally with warnings only.
- Plan sync check: passed.
- Whitespace diff check: passed.
- Local review bundle: pending before push via `scripts/push_pr.sh`.

## Estimated diff size

| File | LOC |
|---|---:|
| `.github/workflows/branch_protection_required_checks.yml` | 53 |
| `.github/workflows/gitleaks_baseline_growth_guard.yml` | 40 |
| `.github/workflows/security_guardrails.yml` | 32 |
| `docs/SECURITY_GUARDRAILS.md` | 18 |
| `plans/INDEX.md` | 1 |
| `plans/PR-Security-Gitleaks-Required-Checks.md` | 200 |
| `plans/archive/PR-Security-Gitleaks-Precommit.md` | 0 |
| `scripts/audit_workflow_security_posture.py` | 2 |
| `scripts/check_required_status_checks.py` | 206 |
| `tests/test_audit_workflow_security_posture.py` | 31 |
| `tests/test_security_guardrails_workflow.py` | 159 |
| **Total** | **742** |

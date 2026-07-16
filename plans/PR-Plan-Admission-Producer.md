# PR-Plan-Admission-Producer

## Why this slice exists

#2104/#2035 call for process gates to move from hidden/advisory checks toward
independently visible branch-protection contexts. #2122 pre-admitted the
`.github/workflows/plan_admission.yml` / `plan-admission` tuple in the
trusted-base workflow-security auditor so the actual producer can now land
without weakening `pre-push-audit`.

This slice adds the producer only. Enrollment as a required branch-protection
context remains a later REST PATCH slice after burn-in and after preserving the
existing required checks.

### Problem-derived contract

- Root cause: `scripts/audit_pr_plan_presence.py` already enforces the
  plan-admission rule locally and inside `pre-push-audit`, but GitHub does not
  expose that rule as its own `plan-admission` context. That makes the process
  gate hard to enroll and inspect independently. #2122 fixed the trusted-base
  bootstrap blocker by pre-admitting the future workflow/job tuple, so the
  remaining missing invariant is the standalone trusted-base producer workflow.
- Correct fix must touch/change: add `.github/workflows/plan_admission.yml`
  using the trusted-base `pull_request_target` pattern; add focused workflow
  tests proving it checks out only the base SHA, materializes the PR head as
  data, and runs the base-owned `scripts/audit_pr_plan_presence.py` against the
  PR data worktree with `ATLAS_AUDIT_REPO_ROOT` pointing at that worktree;
  enroll those workflow tests in `pre-push-audit`; update the pre-push workflow
  regression test to prove enrollment; update `docs/SECURITY_GUARDRAILS.md`
  from future/pre-admitted language to advisory producer language; archive the
  merged #2122 plan by name and refresh `plans/INDEX.md`.
- Must not change: do not alter `scripts/audit_pr_plan_presence.py`
  semantics, docs-only/dependabot exemptions, `session-lane`, `review-contract`,
  `claude-review`, branch protection, required contexts, product behavior,
  workflow-security allowlist semantics, protected S6 work, content-factory
  work, local-model qualification, Dependabot PRs, or dirty files in the main
  checkout.

## Scope (this PR)

Ownership lane: dev-workflow/process-gate-enrollment
Slice phase: Workflow/process

1. Add the standalone trusted-base `plan-admission` GitHub Actions producer.
2. Add workflow tests and CI enrollment proving the producer stays trusted-base
   and data-only.
3. Update security docs to describe the producer as advisory/burn-in rather
   than future/pre-admitted.
4. Move the merged #2122 plan into `plans/archive/` and refresh the plan index.

### Review Contract

- Acceptance criteria:
  - [ ] `.github/workflows/plan_admission.yml` runs only on
        `pull_request_target` and the `plan-admission` job is guarded by
        `if: github.event_name == 'pull_request_target'`.
  - [ ] The workflow checks out the trusted base SHA as executable code before
        any PR data is materialized.
  - [ ] The PR head is fetched into a separate worktree and treated as data.
  - [ ] The audit step runs base-owned
        `scripts/audit_pr_plan_presence.py` from `$GITHUB_WORKSPACE` while the
        current directory is the PR data worktree, sets `ATLAS_AUDIT_REPO_ROOT`
        to the PR data worktree, and passes the base ref and PR author
        explicitly.
  - [ ] The new workflow test is enrolled in both PR and `main`
        `pre-push-audit` tooling-test command lines.
  - [ ] Docs describe `plan-admission` as advisory/burn-in, not required.
  - [ ] No branch-protection or plan-admission semantic change ships here.
  - [ ] The #2122 plan is moved to `plans/archive/` by name and the index is
        refreshed; no bulk archive sweep runs.
- Reachability proof: the real GitHub Actions entrypoint is the new
  `.github/workflows/plan_admission.yml` `pull_request_target` workflow; focused
  tests inspect that workflow and assert the observable command path that runs
  the base-owned plan-admission auditor against the PR data worktree. The
  workflow-security auditor also validates the new workflow's trusted-base
  guard shape.
- Affected surfaces: plan-admission workflow, pre-push workflow test
  enrollment, workflow/security docs, focused workflow tests, and plan archive
  housekeeping.
- Risk areas: executing PR-owned code in a `pull_request_target` context,
  accidentally making `plan-admission` required before burn-in, leaving the new
  workflow tests out of CI, or changing the plan-admission rule semantics.
- Reviewer rules triggered: R1, R2, R3, R10, R12, R14.

### Files touched

- `.github/workflows/plan_admission.yml`
- `.github/workflows/pre_push_audit.yml`
- `docs/SECURITY_GUARDRAILS.md`
- `plans/INDEX.md`
- `plans/PR-Plan-Admission-Producer.md`
- `plans/archive/PR-Plan-Admission-PreAdmission.md`
- `tests/test_plan_admission_workflow.py`
- `tests/test_pre_push_audit_workflow.py`

## Mechanism

The workflow mirrors the already-live `session-lane` trusted-base shape. GitHub
executes the workflow from the base branch, the first step checks out
`${{ github.event.pull_request.base.sha }}` using the SHA-pinned checkout
action, and the PR head is fetched into `$RUNNER_TEMP/pr-tree` only after that
trusted checkout exists.

The audit step then changes directory into the PR data worktree and invokes the
base-owned script by absolute path:
`ATLAS_AUDIT_REPO_ROOT="$RUNNER_TEMP/pr-tree" python
"$GITHUB_WORKSPACE/scripts/audit_pr_plan_presence.py" "origin/${BASE_REF}"
--pr-author "$PR_AUTHOR"`. The environment override is required because the
base-owned script normally derives its inspected repo root from its own script
path. That lets the existing plan-admission rule classify the PR diff without
executing scripts, tests, package code, or workflow code from the PR branch.

## Intentional

- This PR intentionally does not change `scripts/audit_pr_plan_presence.py`; it
  is wiring the existing rule into a visible producer context.
- This PR intentionally does not make `plan-admission` required in branch
  protection; producer burn-in and required-context enrollment are separate
  #2104 steps.
- This PR keeps the workflow self-contained instead of adding another
  `pre-push-audit` mode; the point of the slice is an independently visible
  GitHub context.

## Deferred

- #2104 enrollment slice: require `plan-admission` only after producer burn-in
  and a REST branch-protection patch that preserves every existing context.
- #2104 follow-up producers: add the remaining review-contract and reviewer
  gate producers using the same pre-admit-then-producer sequence.
- #2104 reviewer slice: replace or constrain the forgeable `claude-review`
  publisher before treating it as a hard trust boundary.

Parked hardening: none.

## Verification

- `python -m pytest tests/test_plan_admission_workflow.py tests/test_pre_push_audit_workflow.py tests/test_audit_workflow_security_posture.py -q` — 35 passed.
- `python scripts/audit_workflow_security_posture.py .github/workflows` — passed with existing mutable-action warnings and expected trusted-base allowlist warnings.
- `python scripts/audit_pr_plan_presence.py origin/main --pr-author canfieldjuan` — passed; classified this committed branch as plan-required and found `plans/PR-Plan-Admission-Producer.md`.
- `python scripts/sync_pr_plan.py plans/PR-Plan-Admission-Producer.md --check` — passed.
- `git diff --check` — passed.
- `bash scripts/push_pr.sh /tmp/atlas-plan-admission-producer-pr-body.md -u origin HEAD` — passed; local PR review bundle passed and branch published.

## Estimated diff size

| File | LOC |
|---|---:|
| `.github/workflows/plan_admission.yml` | 54 |
| `.github/workflows/pre_push_audit.yml` | 4 |
| `docs/SECURITY_GUARDRAILS.md` | 14 |
| `plans/INDEX.md` | 3 |
| `plans/PR-Plan-Admission-Producer.md` | 156 |
| `plans/archive/PR-Plan-Admission-PreAdmission.md` | 0 |
| `tests/test_plan_admission_workflow.py` | 38 |
| `tests/test_pre_push_audit_workflow.py` | 1 |
| **Total** | **270** |

# PR-Process-Gate-Required-Status-Audit

## Why this slice exists

#2104 is the follow-up to move the process gates from advisory producer status
to required branch-protection contexts after the trusted-base producers exist.
The three non-reviewer producers now exist on `main`: `session-lane`,
`plan-admission`, and `review-contract`. The remaining repo-side gap is that
the live branch-protection auditor still only knows the old baseline from
`scripts/check_required_status_checks.py:13-18`, while the live payload still
contains only `live-reconciliation`, `Gitleaks PR secret scan`, and `Gitleaks
baseline growth guard`. That means the repo has no watchdog for the newly
enrolled process gates or the already-documented `diff-budget` requirement.

This is a workflow/process slice justified by #2104/#2035: process-gate
enrollment is the safety gap that kept advisory gates bypassable during the
recent non-converging builder arcs.

### Problem-derived contract

- Root cause: the required-status audit's source of truth stops at the legacy
  four-context baseline (`scripts/check_required_status_checks.py:13-18`), so
  it cannot fail when `plan-admission`, `session-lane`, or `review-contract`
  are absent from branch protection. The security docs also still describe the
  new producers as advisory/deferred (`docs/SECURITY_GUARDRAILS.md:80-108`)
  even though they are now on `main` and ready for the non-Claude enrollment
  step.
- Correct fix must touch/change: update
  `scripts/check_required_status_checks.py` to include the non-Claude process
  gate contexts in the default required GitHub Actions check set; update the
  focused tests in `tests/test_security_guardrails_workflow.py` so the default
  pass/fail cases cover those contexts and wrong-source failures; update
  `.github/workflows/branch_protection_required_checks.yml` so the live audit
  is triggered when any producer workflow changes; update
  `docs/SECURITY_GUARDRAILS.md` to describe the required non-Claude context set
  and the separate live REST enrollment step.
- Must not change: do not mutate live branch protection in this PR; do not add
  `claude-review` as a required context; do not change the producer workflow
  semantics for `plan-admission`, `session-lane`, `review-contract`,
  `diff-budget`, Gitleaks, or reconciliation; do not touch product behavior,
  schemas, migrations, extracted packages, customer-facing surfaces, open
  content-factory PR #2126, local-model PR #2117, Dependabot, or protected
  Resolution Audit S6 work.

## Scope (this PR)

Ownership lane: dev-workflow/process-gate-enrollment
Slice phase: Workflow/process

1. Make the branch-protection required-status auditor's default check set match
   the intended non-Claude required contexts.
2. Update the branch-protection audit workflow triggers and security docs so
   producer changes and the live REST enrollment step are visible.
3. Add/update focused tests proving default pass, missing-context fail, legacy
   unpinned fail, and wrong-app fail behavior for the full context set.

### Review Contract

- Acceptance criteria:
  - [ ] The default required check set contains `live-reconciliation`,
        `diff-budget`, `Gitleaks PR secret scan`, `Gitleaks baseline growth
        guard`, `plan-admission`, `session-lane`, and `review-contract`.
  - [ ] Each default context must be pinned to GitHub Actions app id `15368`;
        legacy bare contexts and wrong app IDs still fail.
  - [ ] `claude-review` is intentionally absent from this PR's default set.
  - [ ] The branch-protection audit workflow reruns on edits to all producer
        workflows whose contexts it audits.
  - [ ] Docs no longer describe the now-merged process producers as merely
        future/advisory, and they keep live REST enrollment separate.
- Reachability proof: `scripts/check_required_status_checks.py` run against a
  synthetic full payload exits 0, and against a live-shaped missing payload
  exits 1 in tests/manual verification.
- Affected surfaces: `scripts/check_required_status_checks.py`,
  `.github/workflows/branch_protection_required_checks.yml`,
  `docs/SECURITY_GUARDRAILS.md`, `tests/test_security_guardrails_workflow.py`,
  this plan.
- Risk areas: branch-protection check-name drift, accidentally requiring a
  forgeable/non-GitHub-Actions context, stale docs that claim live protection
  has changed before the REST PATCH, workflow trigger omissions.
- Reviewer rules triggered: R1, R2, R3, R8, R10, R14.

### Files touched

- `.github/workflows/branch_protection_required_checks.yml`
- `docs/SECURITY_GUARDRAILS.md`
- `plans/PR-Process-Gate-Required-Status-Audit.md`
- `scripts/check_required_status_checks.py`
- `tests/test_security_guardrails_workflow.py`

## Mechanism

The auditor already normalizes GitHub's `contexts` and `checks` shapes and
requires each default context to appear in `checks[]` with the GitHub Actions app
id. This PR keeps that mechanism and only expands its default contract to the
non-Claude process-gate contexts that now have merged trusted-base producers.
The branch-protection workflow trigger list is extended so edits to those
producer workflows rerun the live audit. Documentation is updated to separate
"the repo now audits for these contexts" from "the operator/live REST PATCH has
enrolled them."

## Intentional

- `claude-review` is not enrolled here. This follows the current operator policy
  for this arc: Codex/live-reconciliation is the required review signal; Claude's
  commit status remains out of the branch-protection set unless a separate
  reviewer identity decision is made later.
- This PR does not mutate live branch protection. The actual REST PATCH is an
  external settings change after the code/documentation PR is reviewed and
  merged.

## Deferred

- Apply the minimal REST branch-protection PATCH after this PR is reviewed and
  merged: preserve existing checks, add `diff-budget`, `plan-admission`,
  `session-lane`, and `review-contract`, then re-fetch and run the live auditor.
- If the operator later provisions a non-forgeable reviewer identity and wants a
  separate required reviewer status, handle that in a separate policy slice.

Parked hardening: none.

## Verification

- Passed: `python -m pytest tests/test_security_guardrails_workflow.py -q` --
  15 passed.
- Passed: synthetic full-payload CLI probe through
  `scripts/check_required_status_checks.py` -- expected PASS with all seven
  required contexts pinned to GitHub Actions app id `15368`.
- Passed: synthetic current-live-shaped CLI probe through
  `scripts/check_required_status_checks.py` -- expected FAIL on `diff-budget`,
  `plan-admission`, `session-lane`, and `review-contract`.
- Passed: `scripts/push_pr.sh` local-review wrapper -- all local review checks passed.

## Estimated diff size

| File | LOC |
|---|---:|
| `.github/workflows/branch_protection_required_checks.yml` | 7 |
| `docs/SECURITY_GUARDRAILS.md` | 51 |
| `plans/PR-Process-Gate-Required-Status-Audit.md` | 142 |
| `scripts/check_required_status_checks.py` | 3 |
| `tests/test_security_guardrails_workflow.py` | 69 |
| **Total** | **272** |

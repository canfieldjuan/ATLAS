# PR-Plan-Admission-PreAdmission

## Why this slice exists

Issue #2104 calls for plan-admission, `session-lane`, `review-contract`, and
reviewer gates to become independently visible required contexts. #2119 added
the actual `session-lane` producer after #2110 pre-admitted its workflow shape.
The same trusted-base bootstrap applies to the next producer: a PR cannot add a
new `pull_request_target` workflow and its workflow-security allowlist entry in
the same diff, because `pre-push-audit` runs the base branch auditor against PR
workflow files.

This workflow/process slice pre-admits the future plan-admission producer so the
next #2104 slice can add the actual workflow without weakening the trusted-base
guard. It should also be the first real PR after #2119 to exercise the new
advisory `session-lane` context from `main`.

### Problem-derived contract

- Root cause: `scripts/audit_pr_plan_presence.py` already implements plan
  admission locally and under the broad `pre-push-audit` bundle, but there is no
  standalone GitHub context named plan-admission. Adding that future
  `pull_request_target` workflow directly would fail the current trusted-base
  workflow-security audit because base `main` does not yet allowlist the future
  plan-admission workflow/job pair.
- Correct fix must touch/change: pre-admit only the future plan-admission
  workflow/job tuple in `scripts/audit_workflow_security_posture.py`, extend the
  workflow-security
  regression test to cover that exact trusted-base shape, document the bootstrap
  split in `docs/SECURITY_GUARDRAILS.md`, archive the merged #2119 plan by name,
  and refresh `plans/INDEX.md`.
- Must not change: do not add the actual plan-admission workflow, mutate branch
  protection, mark plan-admission required, change plan-admission semantics or
  docs-only/dependabot exemptions, change `session-lane`, `claude-review`, or
  `review-contract`, touch product behavior, touch protected
  S6/content/dependabot/local-model lanes, or weaken `pre-push-audit`.

## Scope (this PR)

Ownership lane: dev-workflow/process-gate-enrollment
Slice phase: Workflow/process

1. Add the future plan-admission workflow/job pair to the workflow-security
   posture allowlist.
2. Add regression coverage proving that the future workflow/job pair is accepted
   only when it keeps the approved trusted-base guard shape.
3. Document that this is pre-admission only; the actual producer and branch
   protection enrollment remain separate #2104 follow-ups.
4. Move the merged #2119 plan into `plans/archive/` and refresh the plan index.

### Review Contract

- Acceptance criteria:
  - [ ] The only runtime code change is the workflow-security allowlist adding
        the future plan-admission workflow/job tuple.
  - [ ] The regression test proves the future workflow/job pair is accepted by
        the same trusted-base guard-shape predicate as existing PR meta-gates.
  - [ ] This PR does not add the future plan-admission workflow file.
  - [ ] Docs describe the context as future/pre-admitted, not currently running
        or required.
  - [ ] The #2119 plan is moved to `plans/archive/` by name and the index is
        refreshed; no bulk archive sweep runs.
  - [ ] The live PR for this slice shows the advisory `session-lane` context
        can run from merged #2119.
- Reachability proof: workflow-security tests exercise the real
  `scripts/audit_workflow_security_posture.py` allowlist and guard-shape
  enforcement. The GitHub PR itself is the first live burn-in candidate for the
  newly merged `session-lane` context.
- Affected surfaces: workflow security posture allowlist/tests, security docs,
  and plan archive housekeeping.
- Risk areas: accidentally loosening the trusted-base predicate, documenting
  plan-admission as required before it exists, or changing plan-admission
  semantics instead of only admitting the future producer shape.
- Reviewer rules triggered: R1, R2, R10, R12, R14.

### Files touched

- `docs/SECURITY_GUARDRAILS.md`
- `plans/INDEX.md`
- `plans/PR-Plan-Admission-PreAdmission.md`
- `plans/archive/PR-Session-Lane-Producer.md`
- `scripts/audit_workflow_security_posture.py`
- `tests/test_audit_workflow_security_posture.py`

## Mechanism

`scripts/audit_workflow_security_posture.py` keeps a tuple allowlist of the only
jobs allowed to run on `pull_request_target`. Every allowlisted job must still
match the exact trusted-base guard shape: job-level
`if: github.event_name == 'pull_request_target'`, a first step using the
SHA-pinned `actions/checkout` action, and checkout ref
`${{ github.event.pull_request.base.sha }}`.

This PR adds only the future plan-admission workflow/job tuple.
The regression test builds a minimal workflow with that file/job name and the
required guard shape, then asserts the real auditor reports no error. A later
#2104 producer PR can add the actual workflow after this allowlist is on `main`.

## Intentional

- This PR intentionally does not add the plan-admission workflow. That split is
  the trusted-base bootstrap fix.
- This PR does not change `scripts/audit_pr_plan_presence.py`; its semantics are
  already the intended producer behavior.
- This PR does not change branch protection or required-check settings.

## Deferred

- #2104 next producer slice: add the actual trusted-base plan-admission workflow
  using `scripts/audit_pr_plan_presence.py`.
- #2104 enrollment slice: require plan-admission only after producer burn-in and
  a REST branch-protection patch that preserves every existing context.
- #2104 reviewer slice: replace or constrain the forgeable `claude-review`
  publisher before requiring it.

Parked hardening: none.

## Verification

- `python -m pytest tests/test_audit_workflow_security_posture.py -q` — 17 passed.
- `python scripts/audit_workflow_security_posture.py .github/workflows` — passed with existing mutable-action warnings and the expected trusted-base allowlist warnings.
- `python scripts/sync_pr_plan.py plans/PR-Plan-Admission-PreAdmission.md --check` — passed after sync.
- `git diff --check` — passed.
- Pending before push: `bash scripts/push_pr.sh /tmp/atlas-plan-admission-preadmission-pr-body.md -u origin HEAD`.

## Estimated diff size

| File | LOC |
|---|---:|
| `docs/SECURITY_GUARDRAILS.md` | 8 |
| `plans/INDEX.md` | 3 |
| `plans/PR-Plan-Admission-PreAdmission.md` | 136 |
| `plans/archive/PR-Session-Lane-Producer.md` | 0 |
| `scripts/audit_workflow_security_posture.py` | 1 |
| `tests/test_audit_workflow_security_posture.py` | 1 |
| **Total** | **149** |

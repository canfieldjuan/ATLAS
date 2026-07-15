# PR-Session-Lane-PreAdmission

## Why this slice exists

Issue #2104 calls for plan, lane, Review Contract, and reviewer gates to move
from advisory process to independently visible required contexts. #2106 landed
the public PR-body `Ownership lane:` metadata needed for a GitHub-side lane
gate, but the next attempt to add the actual `session-lane` workflow exposed a
trusted-base bootstrap constraint: `pre-push-audit` runs the workflow-security
auditor from `main` against the PR's workflow files, so `main` must know a new
`pull_request_target` gate is allowlisted before a PR can add that workflow.

This workflow/process slice therefore pre-admits the future `session-lane`
producer shape. It is the smallest merge-ordered step that lets the next #2104
slice add the future session-lane workflow without weakening the
trusted-base audit or bypassing the red check.

### Problem-derived contract

- Root cause: trusted-base `pre-push-audit` deliberately evaluates PR workflow
  files with the base branch copy of `scripts/audit_workflow_security_posture.py`.
  A PR that adds a new `pull_request_target` workflow and its allowlist entry in
  the same diff still fails, because the base-owned auditor cannot see the PR's
  updated allowlist. The failure is the trust model working correctly, not a
  bug in the future workflow's guard shape.
- Correct fix must touch/change: Land only the base-owned pre-admission pieces:
  add the future session-lane workflow/job pair to the workflow-security allowlist,
  extend the allowlist regression test, document why the producer must land in a
  follow-up after this PR, and carry the #2106 plan archive housekeeping. Do not
  add the actual workflow producer in this PR.
- Must not change: Do not add the actual session-lane workflow, rewrite the
  lane parser, expand metadata grammar, mutate branch protection, enroll
  `session-lane` as required, change `claude-review`, change product behavior,
  touch protected S6/content/dependabot lanes, remove or weaken
  `pre-push-audit`, or optimize Unit Gate runtime in this slice.

## Scope (this PR)

Ownership lane: dev-workflow/process-gate-enrollment
Slice phase: Workflow/process

1. Pre-admit the future session-lane workflow/job pair in
   `scripts/audit_workflow_security_posture.py` under the existing strict
   trusted-base guard-shape requirement.
2. Extend the workflow-security regression test so the future workflow name and
   job pass only with the approved event guard and base-SHA checkout.
3. Document the bootstrapping order: this PR lands the base allowlist; the next
   #2104 producer PR adds the actual workflow and then can pass trusted-base
   `pre-push-audit`.
4. Move the merged Public Lane Contract plan into
   `plans/archive/PR-Public-Lane-Contract.md` and refresh the plan index as
   housekeeping carried forward from #2106 teardown.

### Review Contract

- Acceptance criteria:
  - [ ] The only runtime code change is the workflow-security allowlist adding
        future session-lane workflow/job pair.
  - [ ] The existing guard-shape test proves allowlisting is necessary but still
        insufficient: missing the exact `if` guard or base-SHA checkout remains
        an error.
  - [ ] The PR does not add the actual session-lane workflow.
  - [ ] Docs explain the split as a trusted-base bootstrap requirement and do
        not claim `session-lane` is already a running or required context.
  - [ ] The #2106 plan is moved from `plans/` to `plans/archive/` and the index
        is refreshed; no unrelated plan archive sweep runs.
- Reachability proof: workflow-security tests exercise the real auditor's
  allowlist and guard-shape enforcement. The local pre-push wrapper then runs
  the base-owned workflow-security audit against this PR's workflow files, which
  is the exact failing path from the red CI run.
- Affected surfaces: workflow security posture allowlist/tests, security docs,
  and plan archive housekeeping.
- Risk areas: accidentally loosening the guard shape, documenting a context as
  required before it exists, or sneaking in the actual workflow before base can
  admit it.
- Reviewer rules triggered: R1, R2, R10, R12, R14.

### Files touched

- `docs/SECURITY_GUARDRAILS.md`
- `plans/INDEX.md`
- `plans/PR-Session-Lane-PreAdmission.md`
- `plans/archive/PR-Public-Lane-Contract.md`
- `scripts/audit_workflow_security_posture.py`
- `tests/test_audit_workflow_security_posture.py`

## Mechanism

`scripts/audit_workflow_security_posture.py` keeps its existing hardcoded
trusted-gate allowlist and exact guard-shape check. This PR adds only the future
session-lane workflow/job tuple to that allowlist. The existing tests
already prove that an allowlisted job without the required event guard or
base-SHA checkout is still rejected; this PR extends the positive allowlist case
to include `session-lane`.

Because `pre-push-audit` executes the auditor from trusted base, this
pre-admission must merge before the workflow file can be introduced. The next
producer PR can then add the actual workflow; base `main` will
already recognize that exact workflow/job pair while still enforcing the guard
shape.

## Intentional

- This PR does not add the `session-lane` workflow. That omission is the fix for
  the current red `pre-push-audit` bootstrap failure.
- This PR does not alter the guard-shape predicate. It only pre-admits the
  future workflow/job tuple to the existing predicate.
- The #2106 plan archive is included only as merge-housekeeping for this same
  lane; no bulk archive command or unrelated plan movement is in scope.

## Deferred

- #2104 next producer slice: add the actual session-lane workflow with the
  trusted-base checkout/body/data-worktree pattern now pre-admitted here, but
  keep it advisory or blocked until #2113 closes the fenced Scope metadata
  parser leak.
- #2104 enrollment slice: add `session-lane` to branch protection after the
  producer proves stable on live PRs, #2113 is fixed, and every existing
  required context is preserved.
- #2104 reviewer slice: replace or constrain the forgeable `claude-review`
  status with a distinct reviewer-owned publisher before requiring it.
- Unit Gate speed optimization: logged on #2104 and intentionally left out of
  this pre-admission slice.

Parked hardening: none.

## Verification

- `python -m pytest tests/test_audit_workflow_security_posture.py tests/test_pre_push_audit_workflow.py -q` — 30 passed.
- `python scripts/audit_workflow_security_posture.py .github/workflows` — passed; this is the local equivalent of the red CI step after removing the new workflow file from this pre-admission PR.
- `bash scripts/push_pr.sh /tmp/atlas-session-lane-producer-pr-body.md --force-with-lease origin HEAD:claude/pr-session-lane-producer` — passed; the managed pre-push hook ran `local_pr_review.sh` once with the revised PR body.

## Estimated diff size

| File | LOC |
|---|---:|
| `docs/SECURITY_GUARDRAILS.md` | 19 |
| `plans/INDEX.md` | 3 |
| `plans/PR-Session-Lane-PreAdmission.md` | 143 |
| `plans/archive/PR-Public-Lane-Contract.md` | 0 |
| `scripts/audit_workflow_security_posture.py` | 1 |
| `tests/test_audit_workflow_security_posture.py` | 1 |
| **Total** | **167** |

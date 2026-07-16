# PR-Session-Lane-Producer

## Why this slice exists

Issue #2104 calls for plan, lane, Review Contract, and reviewer gates to move
from advisory process to independently visible required contexts. #2110
pre-admitted the future `session-lane` trusted-base workflow/job shape in
`scripts/audit_workflow_security_posture.py`, and #2115 closed the fenced Scope
metadata parser leak that would have made that gate bypassable.

This workflow/process slice adds the actual `session-lane` producer as an
advisory GitHub Actions context. It is intentionally not branch-protection
enrollment yet: #2104 requires advisory burn-in and separate protection mutation
after the producer exists and proves stable on live PRs.

### Problem-derived contract

- Root cause: the repository now has source-side lane extraction, current-body
  comparison, and workflow-security pre-admission for `session-lane`, but it
  has no independently visible GitHub check named `session-lane`. Builders and
  reviewers can run `scripts/audit_pr_session_drift.py` locally or as part of
  `pre-push-audit`, but branch protection cannot require an absent standalone
  context and a live PR cannot burn in the exact producer.
- Correct fix must touch/change: add `.github/workflows/session_lane.yml` as a
  trusted-base `pull_request_target` producer with job name `session-lane`; have
  it checkout the base SHA, write the PR body from the trusted event payload,
  fetch/materialize the PR head as data, and run the base-owned
  `scripts/audit_pr_session_drift.py --current-pr-body-file ... origin/<base>`
  against the data worktree. Add workflow tests proving the trusted-base shape,
  PR-body handoff, and session-drift command wiring. Run the existing
  workflow-security posture audit so the pre-admitted shape is exercised. Archive
  the merged #2115 plan by name and refresh the plan index.
- Must not change: do not mutate branch protection, mark `session-lane` required,
  add or change `claude-review`, change lane/phase grammar, change the
  session-drift parser semantics that #2115 just fixed, modify product behavior,
  touch protected S6/content/dependabot/local-model lanes, weaken
  `pre-push-audit`, or optimize Unit Gate runtime.

## Scope (this PR)

Ownership lane: dev-workflow/process-gate-enrollment
Slice phase: Workflow/process

1. Add `.github/workflows/session_lane.yml` with the pre-admitted
   `pull_request_target` / trusted-base / PR-head-as-data shape.
2. Wire the new workflow to the existing session-drift auditor and current
   PR-body file comparison.
3. Add tests that keep the workflow shape and command wiring enrolled.
4. Move the merged #2115 plan into `plans/archive/` and refresh the plan index.

### Review Contract

- Acceptance criteria:
  - [ ] The new workflow exposes a `session-lane` job/context on
        `pull_request_target`.
  - [ ] The job checks out trusted base code pinned to
        `github.event.pull_request.base.sha`.
  - [ ] The PR head is fetched/materialized only as git data; no PR-ref scripts,
        tests, package code, or workflow code execute.
  - [ ] The PR body is written from the trusted event payload and passed to
        `scripts/audit_pr_session_drift.py` with `--current-pr-body-file`.
  - [ ] The auditor runs against `origin/${BASE_REF}` from the PR data worktree
        so branch-plan metadata and body metadata are compared through the real
        CLI path.
  - [ ] The workflow-security posture auditor accepts the producer because it
        matches the pre-admitted guard shape.
  - [ ] No branch-protection enrollment or reviewer-status change lands in this
        PR.
- Reachability proof: tests inspect the real workflow file and
  `scripts/audit_workflow_security_posture.py .github/workflows` exercises the
  same workflow-security admission path that runs in `pre-push-audit`.
- Affected surfaces: session-lane GitHub Actions producer, workflow-security
  tests, pre-push tooling-test enrollment if a new workflow test file is added,
  security docs, and plan archive housekeeping.
- Risk areas: accidentally executing PR code under `pull_request_target`,
  wiring the context to a stale/body-less audit path, creating a required-check
  claim before branch protection is intentionally patched, or colliding with
  other sessions' plan archive work.
- Reviewer rules triggered: R1, R2, R10, R12, R14.

### Files touched

- `.github/workflows/pre_push_audit.yml`
- `.github/workflows/session_lane.yml`
- `docs/SECURITY_GUARDRAILS.md`
- `plans/INDEX.md`
- `plans/PR-Session-Lane-Producer.md`
- `plans/archive/PR-Session-Lane-Fence-Leak.md`
- `tests/test_pre_push_audit_workflow.py`
- `tests/test_session_lane_workflow.py`

## Mechanism

The new workflow follows the trusted-base pattern already used by
`.github/workflows/pr_body_contract.yml` and `.github/workflows/pre_push_audit.yml`: it runs only the
`session-lane` job on `pull_request_target`, checks out
`${{ github.event.pull_request.base.sha }}`, writes the PR body to a temporary
file from `github.event.pull_request.body`, fetches the PR head into
`refs/remotes/origin/pr-${PR_NUMBER}`, and adds that ref as a temporary worktree.

The final step runs the base-owned `scripts/audit_pr_session_drift.py` from the
trusted checkout while setting its working directory to the PR data worktree.
That gives the existing auditor the branch diff, branch-added plan doc, open PR
lane collision checks, and current PR-body comparison it already implements,
without executing code from the PR checkout.

## Intentional

- The workflow is producer/advisory only. Branch-protection enrollment is
  deferred to a later #2104 slice after live burn-in.
- The slice reuses `scripts/audit_pr_session_drift.py` rather than introducing a
  second lane parser.
- The workflow keeps `pull-requests: read` and `contents: read` only; no write
  permissions are needed for an advisory check.

## Deferred

- #2104 enrollment slice: add `session-lane` to branch protection only after
  this producer proves stable and the REST protection patch can preserve every
  existing required context.
- #2104 reviewer slice: replace or constrain the forgeable `claude-review`
  status with a distinct reviewer-owned publisher before requiring it.

Parked hardening: none.

## Verification

- `python -m pytest tests/test_session_lane_workflow.py tests/test_audit_workflow_security_posture.py tests/test_pre_push_audit_workflow.py -q` — 35 passed.
- `GITHUB_HEAD_REF=claude/pr-session-lane-producer-advisory python scripts/audit_pr_session_drift.py --skip-github --current-pr-body-file /tmp/atlas-session-lane-producer-pr-body.md --require-current-pr-body` — passed; committed branch diff reports lane `dev-workflow/process-gate-enrollment` and phase `workflow/process`.
- `python scripts/audit_workflow_security_posture.py .github/workflows` — passed with existing mutable-action warnings and the expected `session-lane` trusted-base allowlist warning.
- `python scripts/sync_pr_plan.py plans/PR-Session-Lane-Producer.md --check` — passed.
- `git diff --check origin/main...HEAD` — passed.
- Pending before push: `bash scripts/push_pr.sh /tmp/atlas-session-lane-producer-pr-body.md -u origin HEAD`.

## Estimated diff size

| File | LOC |
|---|---:|
| `.github/workflows/pre_push_audit.yml` | 4 |
| `.github/workflows/session_lane.yml` | 70 |
| `docs/SECURITY_GUARDRAILS.md` | 15 |
| `plans/INDEX.md` | 3 |
| `plans/PR-Session-Lane-Producer.md` | 147 |
| `plans/archive/PR-Session-Lane-Fence-Leak.md` | 0 |
| `tests/test_pre_push_audit_workflow.py` | 6 |
| `tests/test_session_lane_workflow.py` | 38 |
| **Total** | **283** |

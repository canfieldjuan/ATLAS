# PR-Session-Lane-Base-Snapshot

## Why this slice exists

PR #2495 merged while its final `pull_request_target` Session Lane job was
starting. The job refreshed `origin/main` after the squash merge landed, then
reported all 16 PR paths as concurrent base drift. The operator asked for the
diagnosed race to be fixed after run `32875888961` exposed it.

### Problem-derived contract

- Root cause: `.github/workflows/session_lane.yml` checks out the immutable
  `github.event.pull_request.base.sha`, but its materialization step then
  force-fetches the moving base branch into `origin/main`. If the PR merges
  before that fetch, `audit_pr_session_drift.py` compares the PR head against
  the PR's own squash commit and correctly reports self-overlap for the wrong
  comparison snapshot.
- Correct fix must touch/change: pin the local `origin/${BASE_REF}` comparison
  ref to the event's base SHA before the audit, retain the human base branch
  name for GitHub open-PR enumeration, remove the moving-base fetch, and add a
  workflow contract test that proves both the positive snapshot binding and
  the negative absence of the live-branch refspec.
- Must not change: do not weaken `audit_pr_session_drift.py` or its real
  same-file overlap failure, change branch protection, execute PR-owned code in
  the trusted-base workflow, alter other workflows, or touch application APIs,
  schemas, dependencies, or product behavior.

## Scope (this PR)

Ownership lane: dev-workflow/session-lane-admission
Slice phase: Workflow/process

1. Bind Session Lane's local comparison ref to the `pull_request_target` event
   base SHA rather than the later live base branch.
2. Add focused workflow contract proof that the snapshot stays immutable while
   PR-head materialization and named-base GitHub queries remain intact.

### Review Contract

- Acceptance criteria:
  - Across every ordering of event creation, base-branch movement, and audit
    execution, `origin/${BASE_REF}` resolves to
    `github.event.pull_request.base.sha`; settled by
    `tests/test_session_lane_workflow.py::test_session_lane_workflow_pins_comparison_ref_to_event_base_sha`.
  - The workflow does not fetch `refs/heads/${BASE_REF}` into the comparison
    ref, while it still fetches `pull/${PR_NUMBER}/head` strictly as data;
    settled by the same positive/negative workflow contract test.
  - The auditor still receives `origin/${BASE_REF}` so
    `github_base_branch_name()` can enumerate open PRs against the named base;
    settled by
    `tests/test_session_lane_workflow.py::test_session_lane_workflow_passes_current_body_to_base_owned_auditor`.
  - Genuine same-file base movement remains blocking; settled by the unchanged
    `tests/test_audit_pr_session_drift.py::test_cli_fails_when_base_changed_same_file_since_branch_point`.
- Reachability proof: the existing `pull_request_target` Session Lane job is the
  real entrypoint and a zero-exit `session-lane` result is the observable gate.
  Because GitHub executes this trusted-base workflow from `main`, the changed
  producer itself cannot run from this PR; focused workflow-contract tests are
  the pre-merge proof, and the first later PR run is the live post-merge proof.
- Affected surfaces: `.github/workflows/session_lane.yml` and its focused
  workflow contract tests.
- Risk areas: event/base snapshot identity, trusted-base execution, PR-head
  data-only handling, named-base GitHub enumeration, and preservation of real
  concurrent-drift failures.
- Reviewer rules triggered: R1, R2, R3, R5, R8, R10, R12, R13, R14.

### Boundary-change enumeration

Required when this diff changes a guard, validator, normalizer, resolver,
router/classifier, or admission boundary. Name each changed boundary path or
seam in the enumeration; otherwise write "N/A - no boundary change."

- Boundary path/seam: `pull_request_target` event base SHA -> local
  `origin/${BASE_REF}` -> `audit_pr_session_drift.py` comparison ref.
- Replaced-path behaviors: replace the live base-branch force-fetch with a local
  ref update to the already checked-out event base commit.
- Guard-relevant fields: `github.event.pull_request.base.sha`,
  `github.event.pull_request.base.ref`, and `github.event.pull_request.number`.
- Caller x input shape: Session Lane receives an immutable 40-hex base SHA, a
  GitHub branch name, and a numeric PR number; only the PR head is materialized
  as untrusted data.

### Deployed-config probing

Required for guard, validator, resolver, admission-boundary, or env/config
fallback changes; otherwise write "N/A - no guard/config boundary change."

- Deployed/default config values: the workflow derives all three inputs from the
  `pull_request_target` event; there is no repository-configured fallback.
- Explicit value probe: the workflow contract binds `BASE_SHA` to
  `${{ github.event.pull_request.base.sha }}` and uses it in `git update-ref`.
- Absent value probe: N/A - GitHub's pull-request event contract supplies the
  base SHA/ref and PR number; the workflow has no alternate event path.
- Default-session/default-context probe: the job-level event guard remains
  `github.event_name == 'pull_request_target'`.
- Side-effect ordering: materialize the PR commit object, pin the comparison
  ref, add the PR worktree, then invoke the base-owned auditor.

### Files touched

- `.github/workflows/session_lane.yml`
- `plans/PR-Session-Lane-Base-Snapshot.md`
- `tests/test_session_lane_workflow.py`

## Mechanism

Expose the event base SHA to the materialization step. Fetch only the PR-head
ref from GitHub, then use `git update-ref` to bind the local
`refs/remotes/origin/${BASE_REF}` to the event base commit already present from
the trusted checkout. The existing auditor can keep receiving `origin/main`:
its Git diff sees the immutable snapshot, while its GitHub query still derives
the base branch name `main`.

## Intentional

- Keep the auditor's same-file overlap semantics unchanged; suppressing overlap
  when the base contains the PR would hide genuine drift and treat the symptom.
- Do not pass the raw base SHA as the auditor's only `base_ref`, because the
  auditor also derives the GitHub base branch name from that argument.
- Keep PR-head checkout data-only and continue executing workflow logic from the
  trusted event base.

## Deferred

The first later PR after merge provides live trusted-base reachability evidence;
the current PR cannot execute its own `pull_request_target` workflow definition.

Parked hardening: none.

## Verification

- Expected red: the new snapshot regression failed because the workflow had no
  `BASE_SHA` event binding (`1 failed`).
- `./ops test focused tests/test_session_lane_workflow.py -q` - `3 passed`.
- `./ops test focused tests/test_audit_pr_session_drift.py::test_cli_fails_when_base_changed_same_file_since_branch_point -q` - `1 passed`; real
  same-file base drift remains blocking.
- Boundary probe: event SHA binding and ref-update-before-worktree pass; the
  moving base-branch refspec is absent; genuine overlap rejection still passes.
- Effect trace against PR #2495's final head: the live merged `origin/main`
  produces `16` overlapping paths, while the event base SHA produces `0`.
- Workflow YAML parsed successfully with the expected `Session Lane` name.
- Plan sync, PR-body audit, and `git diff --check` passed; the mechanical push
  review remains pending.
- Full unit gate: GitHub-only; do not run locally.

## Estimated diff size

| File | LOC |
|---|---:|
| `.github/workflows/session_lane.yml` | 3 |
| `plans/PR-Session-Lane-Base-Snapshot.md` | 152 |
| `tests/test_session_lane_workflow.py` | 11 |
| **Total** | **166** |

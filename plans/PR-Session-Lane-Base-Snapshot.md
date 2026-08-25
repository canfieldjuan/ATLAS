# PR-Session-Lane-Base-Snapshot

## Why this slice exists

PR #2495 merged while its final `pull_request_target` Session Lane job was
starting. The job refreshed `origin/main` after the squash merge landed, then
reported all 16 PR paths as concurrent base drift. The first PR #2496 head
froze the comparison at the event base; current-head review proved that this
also hides legitimate unrelated base changes that land after event creation.

### Problem-derived contract

- Root cause: the workflow applies a moving-base drift audit without first
  distinguishing an `OPEN` PR, where new base commits remain relevant, from a
  `CLOSED` or `MERGED` PR, where the gate can no longer affect admission and the
  current PR's own squash commit may be present on the base. Freezing the event
  base treats the symptom and suppresses real post-event drift.
- Correct fix must touch/change: retain the live base fetch; after that snapshot
  is fixed locally, query the current PR state and run the auditor only for
  `OPEN`; treat canonical terminal states as a successful no-op; fail unknown
  or malformed states; and add workflow contract tests for both open/live-drift
  and terminal/self-merge timing directions.
- Must not change: do not weaken `audit_pr_session_drift.py` or its real
  same-file overlap failure, change branch protection, execute PR-owned code in
  the trusted-base workflow, alter other workflows, or touch application APIs,
  schemas, dependencies, or product behavior.

### Contract revision 1

- New evidence: the event-base pin makes `base_files` empty when an unrelated
  PR lands after event creation, so the current PR can miss a genuine overlap
  after that other PR leaves the open-PR set.
- Revised root cause: PR lifecycle state, not base snapshot age, decides whether
  the live drift audit is still meaningful.
- Revised required change surface: restore the live-base refspec, then gate the
  audit on a live `gh pr view` state query performed after the fetch; update the
  workflow contract tests, boundary closure declaration, parking predicate,
  and reconciliation record.
- Revised non-scope: keep `audit_pr_session_drift.py` and its real overlap
  semantics unchanged; do not add a second drift algorithm or touch other CI.
- Revised verification plan: expected-red tests for the open/terminal state
  gate, focused workflow tests, unchanged real-overlap proof, YAML/plan/body/
  diff audits, and the mechanical push. Full units remain GitHub-only.

## Scope (this PR)

Ownership lane: dev-workflow/session-lane-admission
Slice phase: Workflow/process
Max files: 3

1. Snapshot the live base and PR head, then audit only while the current PR is
   still open.
2. Add focused workflow contract proof for open/live-drift, terminal/self-merge,
   unknown-state failure, and preserved PR-head data-only handling.

### Review Contract

- Acceptance criteria:
  - Across every ordering of event creation, unrelated base movement, current
    PR closure, and audit execution, the workflow first snapshots the live base
    and then runs the audit exactly when the live PR state is `OPEN`; settled by
    `tests/test_session_lane_workflow.py::test_session_lane_workflow_snapshots_live_base_before_state_gate`.
  - Canonical `CLOSED` and `MERGED` states exit successfully before the auditor,
    while any unrecognized state fails; settled by
    `tests/test_session_lane_workflow.py::test_session_lane_workflow_closes_pr_state_class`.
  - The workflow still fetches `pull/${PR_NUMBER}/head` strictly as data and
    never executes code from its worktree; settled by the live-base/state-gate
    test plus the trusted-base workflow test.
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
- Risk areas: fetch/state-check ordering, terminal-state no-op behavior,
  unknown-state failure, trusted-base execution, PR-head data-only handling,
  named-base GitHub enumeration, and preservation of real concurrent drift.
- Reviewer rules triggered: R1, R2, R3, R5, R8, R10, R12, R13, R14.

### Boundary-change enumeration

Required when this diff changes a guard, validator, normalizer, resolver,
router/classifier, or admission boundary. Name each changed boundary path or
seam in the enumeration; otherwise write "N/A - no boundary change."

- Boundary path/seam: live base ref + live PR lifecycle state -> terminal-state
  gate -> `audit_pr_session_drift.py` comparison ref.
- Replaced-path behaviors: replace event-base pinning with a live-base snapshot
  followed by a closed PR-state decision before audit execution.
- Guard-relevant fields: `github.event.pull_request.base.ref`,
  `github.event.pull_request.number`, and the live GitHub pull-request state.
- Caller x input shape: Session Lane receives a GitHub branch name and numeric
  PR number, materializes only the PR head as untrusted data, and receives one
  canonical live PR-state enum value from `gh pr view`.

### Closure declaration

- Set: pull-request lifecycle states that decide whether Session Lane executes
  the drift auditor.
- Membership: CLOSED; GitHub's pull-request state contract supplies `OPEN`,
  `CLOSED`, or `MERGED`, so unlisted canonical members are impossible.
- Source: ENUMERATED in the workflow's case decision from GitHub's canonical
  `PullRequestState` values, with the current member selected by live
  `gh pr view <number> --json state` output at the decision point.
- Outside-set behavior: query failure, empty output, or an unrecognized state
  fails the job under `set -euo pipefail`/the explicit default branch. `OPEN`
  audits the live base; `CLOSED` and `MERGED` exit successfully because the gate
  can no longer change admission.

### Deployed-config probing

Required for guard, validator, resolver, admission-boundary, or env/config
fallback changes; otherwise write "N/A - no guard/config boundary change."

- Deployed/default config values: the workflow derives the base ref and PR
  number from the `pull_request_target` event and lifecycle state from GitHub;
  there is no repository-configured fallback.
- Explicit value probe: workflow contract tests cover the `OPEN`, `CLOSED`,
  `MERGED`, and default state branches plus the live-base refspec.
- Absent value probe: N/A - GitHub's pull-request event contract supplies the
  base SHA/ref and PR number; the workflow has no alternate event path.
- Default-session/default-context probe: the job-level event guard remains
  `github.event_name == 'pull_request_target'`.
- Side-effect ordering: snapshot live base and PR head, create the PR worktree,
  read the current PR state, then invoke the base-owned auditor only for `OPEN`.

### Files touched

- `.github/workflows/session_lane.yml`
- `plans/PR-Session-Lane-Base-Snapshot.md`
- `tests/test_session_lane_workflow.py`

## Mechanism

Fetch the live base ref and PR head as one materialization step. In the later
base-owned audit step, query the current PR lifecycle state. `OPEN` executes the
existing auditor against the fetched live base; `CLOSED` and `MERGED` exit zero
before the audit because admission is already terminal; any other state fails.
Since the fetch precedes the state query, an open PR retains all unrelated
post-event base movement, while a PR whose own merge entered that snapshot is
recognized as terminal rather than misclassified as drift.

## Intentional

- Keep the auditor's same-file overlap semantics unchanged; event-base pinning
  and overlap suppression both hide genuine drift and are rejected.
- Query current PR state only after the live fetch so every `OPEN` decision uses
  a fixed comparison snapshot that contains all base movement visible then.
- Keep PR-head checkout data-only and continue executing workflow logic from the
  trusted event base.

## Deferred

The first later PR after merge provides live trusted-base reachability evidence;
the current PR cannot execute its own `pull_request_target` workflow definition.

Parking predicate: adjacent workflow hardening is parked unless current
evidence proves it is required to preserve live drift detection, avoid
closed-PR self-overlap, enforce the declared state boundary, or maintain trusted
base execution.

Parked hardening: none.

## Verification

- Earlier expected red: the initial snapshot regression failed because the
  workflow had no `BASE_SHA` event binding (`1 failed`). Current-head review
  invalidated that mechanism because it suppressed post-event drift.
- Revised expected red: the workflow contract file reported `2 failed, 2
  passed` because the current head neither refreshed the live base nor closed
  the PR-state class.
- `./ops test focused tests/test_session_lane_workflow.py -q` - `4 passed`.
- `./ops test focused tests/test_audit_pr_session_drift.py::test_cli_fails_when_base_changed_same_file_since_branch_point -q` - `1 passed`; real
  same-file base drift remains blocking.
- Boundary/effect probe: PR #2496 returns canonical state `OPEN` and reaches the
  live-base audit path; merged PR #2495 returns `MERGED` and reaches the terminal
  no-op path; the explicit default branch rejects unknown/empty state.
- Plan sync, PR-body audit, fix-loop disposition audit, YAML parse, and `git
  diff --check` pass after the revision.
- Mechanical push review remains pending and will run once through
  `scripts/push_pr.sh`.
- Full unit gate: GitHub-only; do not run locally.

## Estimated diff size

| File | LOC |
|---|---:|
| `.github/workflows/session_lane.yml` | 15 |
| `plans/PR-Session-Lane-Base-Snapshot.md` | 198 |
| `tests/test_session_lane_workflow.py` | 33 |
| **Total** | **246** |

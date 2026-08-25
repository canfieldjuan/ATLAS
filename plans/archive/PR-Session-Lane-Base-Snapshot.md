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

### Contract revision history

- Revision 1 — New evidence: event-base pinning hides later unrelated drift. Revised root: lifecycle state decides whether live drift is meaningful. Required surface: restore the live ref and gate audit on current PR state. Non-scope: auditor semantics and other CI. Verification: focused state tests, real-overlap proof, and mechanical audits; full units remain GitHub-only.
- Revision 2 — New evidence: YAML-fragment assertions do not execute the lifecycle decision. Revised root: branch placement was unproved. Required surface: run the base-owned audit-step shell with stubbed `gh`/auditor boundaries for `OPEN`, `CLOSED`, `MERGED`, and unknown state. Non-scope: runtime helper extraction, workflow changes, dependencies, and auditor changes. Verification: focused execution proof and mechanical audits only.
- Revision 3 — New evidence: an overlapping PR can merge after the base snapshot but before open-PR enumeration, disappearing from both evidence sources. Revised root: no linearization check proves the audited base stayed stable. Required surface: compare the audited OID with a post-audit remote point read and exercise stable/moved outcomes. Non-scope: auditor changes, retries, locks, merge-queue/branch-policy provisioning, and other workflows. Verification: expected-red interleaving probe, focused workflow/real-overlap tests, and mechanical audits; full units remain GitHub-only.

## Scope (this PR)

Ownership lane: dev-workflow/session-lane-admission
Slice phase: Workflow/process
Max files: 3

1. Snapshot the live base and PR head, audit only while the current PR is open,
   and allow success only if a post-audit remote OID read proves that base
   snapshot stayed stable through validation.
2. Add executable focused workflow contract proof for open/live-drift,
   terminal/self-merge, unknown-state failure, and preserved PR-head data-only
   handling.

### Review Contract

- Acceptance criteria:
  - For every base-ref movement admitted between the live fetch and final remote
    OID validation, an `OPEN` job returns success only when the audited OID still
    equals the remote OID; settled by
    `tests/test_session_lane_workflow.py::test_session_lane_workflow_rejects_base_movement_during_audit`.
  - Canonical `CLOSED` and `MERGED` states exit successfully before the auditor,
    while any unrecognized state fails; settled by
    `tests/test_session_lane_workflow.py::test_session_lane_workflow_executes_lifecycle_gate`.
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
  unknown-state failure, fetch-to-validation base movement, trusted-base
  execution, PR-head data-only handling, named-base GitHub enumeration, and
  preservation of real concurrent drift.
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

### Execution model (3k.4)

- Selected component and guarantee: native Git commit identity supplies an
  immutable audited OID, and `git ls-remote` supplies an authoritative remote-ref
  point read. Equality closes base movement during this job without inventing a
  second overlap algorithm.
- Model and linearization point: call the fetched OID `S0`; for an `OPEN` PR,
  the existing auditor evaluates `S0`, then the workflow reads remote OID `S1`.
  The successful job linearizes at that final read and admits success only when
  `S0 == S1`.
- Invariant: under every interleaving admitted from initial fetch through final
  validation, any base merge after `S0` is captured either in `S0` itself or by
  `S1 != S0`; it cannot coexist with a successful status.
- Explicit assumption: base movement after the final point read is later than
  this job's execution window and remains subject to GitHub's repository-level
  admission policy; this slice does not claim atomicity from status success to
  PR merge.
- Rejected component: a GitHub merge queue or strict up-to-date branch policy
  would close status-to-merge admission but is external repository policy, not a
  code-local correction to this workflow race. This slice therefore uses one
  fail-closed validation surface and does not add a retry state machine.

### Deployed-config probing

Required for guard, validator, resolver, admission-boundary, or env/config
fallback changes; otherwise write "N/A - no guard/config boundary change."

- Deployed/default config values: the workflow derives the base ref and PR
  number from the `pull_request_target` event and lifecycle state from GitHub;
  there is no repository-configured fallback.
- Explicit value probe: the focused test executes the audit-step shell for
  `OPEN`, `CLOSED`, `MERGED`, and an unknown state and separately asserts status
  plus auditor invocation; structural checks cover the live-base refspec.
- Absent value probe: N/A - GitHub's pull-request event contract supplies the
  base SHA/ref and PR number; the workflow has no alternate event path.
- Default-session/default-context probe: the job-level event guard remains
  `github.event_name == 'pull_request_target'`.
- Side-effect ordering: snapshot live base and PR head, create the PR worktree,
  read the current PR state, invoke the base-owned auditor only for `OPEN`, then
  point-read the remote base OID and permit success only if it stayed stable.

### Files touched

- `.github/workflows/session_lane.yml`
- `plans/PR-Session-Lane-Base-Snapshot.md`
- `tests/test_session_lane_workflow.py`

## Mechanism

Fetch the live base ref and PR head as one materialization step. In the later
base-owned audit step, query the current PR lifecycle state. `OPEN` executes the
existing auditor against the fetched live base; `CLOSED` and `MERGED` exit zero
before the audit because admission is already terminal; any other state fails.
After a successful open-PR audit, compare the immutable audited OID with a fresh
remote-ref point read and fail if they differ. Thus an open PR retains all base
movement present at the initial snapshot and cannot pass when the base moves
during the audit window, while a PR whose own merge entered the initial snapshot
is recognized as terminal rather than misclassified as drift.

## Intentional

- Keep the auditor's same-file overlap semantics unchanged; event-base pinning
  and overlap suppression both hide genuine drift and are rejected.
- Query current PR state only after the live fetch so every `OPEN` decision uses
  a fixed comparison snapshot that contains all base movement visible then.
- Keep PR-head checkout data-only and continue executing workflow logic from the
  trusted event base.
- Fail closed on any fetch-to-validation base movement rather than adding a
  retry loop or changing repository-level merge admission policy.

## Deferred

The first later PR after merge provides live trusted-base reachability evidence;
the current PR cannot execute its own `pull_request_target` workflow definition.

Parking predicate: adjacent workflow hardening is parked unless current
evidence proves it is required to preserve live drift detection, avoid
closed-PR self-overlap, enforce the declared state boundary, or maintain trusted
base execution.

Parked hardening: none.

## Verification

- Expected-red history: event pin absent (`1 failed`); live-base/state gate absent (`2 failed, 2 passed`); post-fetch movement admitted (`1 failed, 7 passed`).
- Focused proof progression: text-only lifecycle proof (`4 passed`); executable lifecycle proof (`7 passed`); final lifecycle plus base-stability proof (`8 passed`).
- Final focused command: `./ops test focused tests/test_session_lane_workflow.py -q` — `8 passed`; stable OIDs permit success, while a moved remote OID fails after exactly one audit.
- `./ops test focused tests/test_audit_pr_session_drift.py::test_cli_fails_when_base_changed_same_file_since_branch_point -q` - `1 passed`; real
  same-file base drift remains blocking.
- Boundary/effect probe: executing the real audit-step shell proves `OPEN`
  reaches the auditor and post-audit OID validation, `CLOSED`/`MERGED` are
  terminal zero-exit no-ops, unknown state never invokes the auditor, and any
  fetch-to-validation base movement fails after the audit instead of passing.
- Plan sync, PR-body audit, fix-loop disposition audit, YAML parse, and `git
  diff --check` pass after the revision.
- Mechanical push review remains pending and will run once through
  `scripts/push_pr.sh`.
- Full unit gate: GitHub-only; do not run locally.

## Estimated diff size

| File | LOC |
|---|---:|
| `.github/workflows/session_lane.yml` | 21 |
| `plans/PR-Session-Lane-Base-Snapshot.md` | 215 |
| `tests/test_session_lane_workflow.py` | 143 |
| **Total** | **379** |

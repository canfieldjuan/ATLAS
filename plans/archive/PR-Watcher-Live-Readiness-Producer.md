# PR-Watcher-Live-Readiness-Producer

## Why this slice exists

The operator asked to continue the Atlas overnight-PR safety arc after
`PR-Watcher-Zero-Thread-Readiness` merged as #2062. That predecessor made the
wake bridge and reporter reject incomplete readiness snapshots, but it
intentionally left the only scheduled producer as an untracked
`~/.local/bin/atlas-pr-watch` file. The live file still summarizes all checks
without discovering the required subset, counts comments/reviews without
fetching review threads, reads the PR head only once, and never emits the
version-1 proof. The consumers are now safely fail-closed, but no scheduled
watcher can truthfully reach ready.

This workflow/process slice is justified by a release-gate safety risk and the
real failed reachability left by #2062: a machine-local producer outside CI can
silently drift from the merge-readiness contract. Diff-budget override: the
installed-runtime slice exceeds the 400-LOC target because the producer,
adversarial GitHub transport fixtures, installer reachability, documentation,
and dedicated CI enrollment are one indivisible trust boundary. Splitting them
would ship either uninstalled floating code or an installed release gate
without failure-branch proof.

### Problem-derived contract

- Root cause: the scheduled watcher executable is machine-local rather than
  repository-owned, so source review and CI cannot enforce how it discovers
  GitHub state or whether its `ready_for_human_merge` label is backed by the
  version-1 proof. Its single head read also leaves a check-collection race.
- Correct fix must touch/change: add one repo-owned one-shot watcher producer;
  install and verify that exact source through the existing wake-bridge
  installer; fetch all checks plus the actual required subset, paginate every
  review-thread page, retain unresolved outdated threads, re-read the head
  after collection, run reconciliation from an installed trusted copy, and
  atomically write the existing snapshot with the version-1 proof; make every
  API, schema, pagination,
  required-set, head-race, review, and merge-state failure non-ready; enroll
  the installed entrypoint and negative fixtures in dedicated CI.
- Must not change: do not add GitHub mutation or merge commands, do not change
  the version-1 consumer contract or wake classifications from #2062, do not
  change AI-reconciliation semantics, do not mutate branch protection, do not
  modify open PR #2063 or its foreground shell watcher, and do not touch any
  product/runtime lane.

## Scope (this PR)

Ownership lane: workflow/pr-watcher-readiness
Slice phase: Workflow/process

Max files: 7

1. Replace the untracked scheduled producer with a repository-owned installed
   copy that preserves the current config/state/session-log interface while
   emitting the version-1 readiness proof.
2. Discover required checks separately from the all-check attention summary,
   fetch every review-thread page, and double-read the PR head so incomplete,
   malformed, stale, or contradictory observations fail closed.
3. Prove the real installed CLI writes a consumer-accepted snapshot and enroll
   every changed watcher/installer test in the dedicated PR-head job.

### Review Contract

- Acceptance criteria:
  - [ ] The existing installer writes and `--check` verifies an executable
        `atlas-pr-watch` byte-for-byte from repository source; its wrapper calls
        that installed path rather than assuming `~/.local/bin`.
  - [ ] The installer writes and verifies the reconciliation checker plus its
        parser dependency, and the watcher never executes reconciliation code
        from the watched PR worktree.
  - [ ] A one-shot poll fetches the same open, non-draft head before and after
        collection, branch protection's required-context inventory, all check
        runs, the currently reported required runs, every review-thread page,
        the current review decision/merge state, and live reconciliation before
        it can label the snapshot ready.
  - [ ] Only required checks with pass buckets are complete/green; empty,
        unreported, pending, failed, canceled, skipped, unknown, or malformed
        required results/policy cannot produce ready.
  - [ ] Thread pagination is complete and bounded; every unresolved thread,
        including outdated threads, appears in the proof and blocks ready.
  - [ ] GitHub command failures, GraphQL `errors`, malformed envelopes, missing
        cursors, page-cap exhaustion, changed heads, dirty worktrees, drafts,
        changes requested, and non-clean merge states all fail closed with
        diagnosable snapshot fields or a non-zero producer error.
  - [ ] Snapshot replacement is atomic, and untrusted PR text written to the
        session-state/log receipt cannot inject multiline control text.
  - [ ] New review/comment activity produces `review_changed` even when checks
        are pending, so attention events are not silently swallowed.
  - [ ] The dedicated workflow path filters and pytest command execute the new
        producer tests plus every modified installer test.
- Reachability proof: install into temporary bin/systemd directories, execute
  the installed `atlas-pr-watch` entrypoint against a fake outermost `gh`
  transport and real config/state filesystem, then assert the observable JSON
  is accepted by `scripts/codex_wake_bridge.py::readiness_blockers`.
- Affected surfaces: developer tooling, GitHub CLI transport, local watcher
  installation, scheduled watcher state, CI enrollment, operator handoff docs.
- Risk areas: false-green merge readiness, head/check races, incomplete
  pagination, untrusted GitHub text, atomic state publication, installer drift,
  backward compatibility with existing watcher configs.
- Reviewer rules triggered: R1, R2, R3, R6, R7, R8, R10, R12, R14.

### Files touched

- `.github/workflows/codex_wake_bridge_checks.yml`
- `docs/long_running_session_watcher_handoff.md`
- `plans/PR-Watcher-Live-Readiness-Producer.md`
- `scripts/install_codex_wake_bridge.py`
- `scripts/pr_watcher.py`
- `tests/test_install_codex_wake_bridge.py`
- `tests/test_pr_watcher.py`

## Mechanism

`scripts/pr_watcher.py` is a standalone one-shot producer copied to
`atlas-pr-watch` by `scripts/install_codex_wake_bridge.py`. It loads the
existing per-session config, performs argument-vector-only `gh` calls, and
keeps all GitHub access behind the subprocess transport. It reads PR metadata,
branch protection's required-context policy, all checks, currently reported
required checks, flat review activity, and paginated GraphQL review threads;
runs the installed trusted reconciliation checker; reads PR metadata again; and
classifies only a stable observation. Comparing policy contexts with reported
required runs prevents a required job that has not materialized from vanishing
from the proof. The proof stores the initial evaluated head, required-check
results, complete pagination receipt, unresolved thread descriptors without
comment bodies, review decision, and merge state.

Pending check runs remain `pending`; API/schema/head/required/thread/review or
merge blockers become `attention`; a closed PR becomes `closed`; only a proof
that the #2062 consumer predicate accepts becomes
`ready_for_human_merge`. JSON is written through a same-directory temporary
file plus `os.replace`, so concurrent reporters never see half a snapshot.
Human-readable receipts collapse GitHub-controlled fields to one line before
updating the local session state/log.

The existing installer copies the watcher, bridge, reconciliation checker and
parser dependency, and wrapper as one versioned installation and verifies
exact content plus executability. The
installed-entrypoint test proves the actual executable path reaches a
version-1 state file.

## Intentional

- The version-1 proof contract from #2062 is unchanged. `claude-review` is not
  invented as an "actual required" check while branch protection does not
  require it; AGENTS live merge guards still require that separate review gate
  before any active builder merge.
- Skipped required checks fail closed even if GitHub might otherwise consider
  a skipped job satisfied. A release gate that did not execute is not evidence
  of green behavior.
- The GraphQL proof stores thread identifiers/locations/resolution flags, not
  comment bodies. Bodies are unnecessary for a zero-thread invariant and are
  untrusted prompt material.
- Open PR #2063 owns a foreground overnight status script. This slice does not
  edit or depend on that branch; the installed systemd JSON producer and the
  foreground watcher remain separate entrypoints with separate state duties.

## Deferred

- Global branch-protection changes remain operator-owned: enable required
  conversation resolution after parallel lanes are assessed; reconcile the
  live required set with repo policy (`diff-budget` is currently absent); and
  require `claude-review` only after a distinct reviewer identity exists.
- Central repo-only safety-audit enrollment for `scripts/pr_watcher.py` is a
  follow-up after open PR #2063 lands. That PR already owns the exact
  `REPO_WATCHER_SOURCES` and `build_findings` hunks required to establish the
  shared registry, so this slice must not create a conflicting second owner.
- A true review-event-to-Codex wake integration remains outside this producer;
  it records scheduled state and the existing bridge builds the handoff.

Parked hardening: none.

## Verification

- `python -m pytest tests/test_codex_wake_bridge.py tests/test_codex_issue_queue.py tests/test_install_codex_wake_bridge.py tests/test_pr_watcher.py tests/test_report_pr_watcher_state.py tests/test_audit_pr_watcher_safety.py -q` - 152 passed.
- `python scripts/maturity_sweep.py scripts --tests-root tests --baseline tests/maturity_sweep/baseline_scripts.json --min-score 8 --sensitive-glob 'scripts/**'` - ratchet passed; the new producer scores 4 with no new brittleness above baseline.
- Temporary `install_codex_wake_bridge.py` install + `--check`, installed watcher/bridge/reconciliation-checker `--help`, and installed-source watcher safety audit - passed.
- `python scripts/audit_pr_watcher_safety.py --repo-root . --repo-only` - passed for the audit's current registered surfaces.
- `python scripts/audit_workflow_security_posture.py .github/workflows` - passed with pre-existing warnings only.
- `python scripts/sync_pr_plan.py plans/PR-Watcher-Live-Readiness-Producer.md --check` - passed.
- `git diff --check` and focused `py_compile` - passed.
- Pending before push: the single `push_pr.sh` local-review run.

## Estimated diff size

| File | LOC |
|---|---:|
| `.github/workflows/codex_wake_bridge_checks.yml` | 3 |
| `docs/long_running_session_watcher_handoff.md` | 45 |
| `plans/PR-Watcher-Live-Readiness-Producer.md` | 193 |
| `scripts/install_codex_wake_bridge.py` | 77 |
| `scripts/pr_watcher.py` | 743 |
| `tests/test_install_codex_wake_bridge.py` | 80 |
| `tests/test_pr_watcher.py` | 773 |
| **Total** | **1914** |

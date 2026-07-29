# PR-Watcher-Zero-Thread-Readiness

## Why this slice exists

The operator asked to close the Atlas CI gaps found while designing a
non-continuous overnight PR loop. The current repo-owned wake bridge and
ready-state reporter can promote a local snapshot that merely says
`ready_for_human_merge`, even though the untracked local producer does not
prove required-check completion, complete review-thread pagination, zero
unresolved threads, review-decision state, or clean mergeability. That is a
release-gate safety defect: an incomplete snapshot can be presented as merge
ready.

Diff-budget override: the 557-LOC plan/test/code/doc slice is indivisible at
the safety boundary. Splitting the shared predicate from either readiness
consumer, its failure-branch fixtures, or dedicated CI enrollment would leave
one false-ready path unguarded or ship a release gate whose negative branches
are not executed by CI.

### Problem-derived contract

- Root cause: the repo-owned readiness consumers trust an unversioned state
  label instead of requiring the evidence that makes the label true. The
  local watcher is outside repository source and can therefore omit or
  miscompute merge-critical inputs without the bridge/reporter CI noticing.
- Correct fix must touch/change: define one fail-closed readiness-proof
  contract in the installed wake-bridge source, require it before a scheduled
  wake or reporter entry is classified ready, expose every failed predicate in
  the handoff/report, add two-sided and contradictory-state fixtures, and run
  both consumer test files in the dedicated PR-head workflow.
- Must not change: do not add GitHub polling or merge commands to the bridge,
  do not grant watcher merge authority, do not modify machine-local watcher or
  systemd files, do not change AI-reconciliation semantics, and do not touch
  product/runtime lanes.

## Scope (this PR)

Ownership lane: workflow/pr-watcher-readiness
Slice phase: Workflow/process

1. Require a versioned snapshot proof tied to the PR head before
   `ready_for_human_merge` can become a scheduled-ready wake or ready reporter
   entry.
2. Fail closed and surface reasons for missing, malformed, stale, pending,
   unresolved-thread, changes-requested, draft/closed, or unmergeable proof.
3. Enroll the bridge and reporter boundary fixtures in the dedicated wake CI.

### Review Contract

- Acceptance criteria:
  - [ ] A complete proof for the same open, non-draft PR head, with at least
        one required check, no failed/pending required checks, complete thread
        pagination, zero unresolved threads, no changes-requested decision,
        and clean merge state is the only snapshot that reports ready.
  - [ ] Missing or contradictory proof becomes attention and lists the exact
        blockers; it never launches a scheduled-ready command.
  - [ ] Outdated-but-unresolved threads still block because the unresolved
        list is non-empty regardless of thread age.
  - [ ] Event wakes remain attention-only and the bridge remains unable to
        poll, mutate, or merge GitHub state.
  - [ ] The dedicated PR-head workflow runs both changed consumer test files.
- Reachability proof: run the dedicated workflow's exact pytest command
  locally; workflow path filters and test list include every changed guard.
- Affected surfaces: developer tooling, local watcher handoff, CI enrollment.
- Risk areas: false-green release readiness, stale head state, incomplete
  pagination, backward compatibility for legacy local snapshots.
- Reviewer rules triggered: R1, R2, R10, R12, R14.

### Files touched

- `.github/workflows/codex_wake_bridge_checks.yml`
- `docs/long_running_session_watcher_handoff.md`
- `plans/PR-Watcher-Zero-Thread-Readiness.md`
- `scripts/codex_wake_bridge.py`
- `scripts/report_pr_watcher_state.py`
- `tests/test_codex_wake_bridge.py`
- `tests/test_report_pr_watcher_state.py`

## Mechanism

`scripts/codex_wake_bridge.py` owns a pure `readiness_blockers` predicate so
the installed standalone bridge carries the contract with it. A version-1
`readiness` object must attest the evaluated head, required-check result,
review-thread pagination/result, review decision, and merge state; duplicated
PR metadata is cross-checked rather than trusted independently. The bridge
uses the predicate before emitting `scheduled-ready` and records blocker text
in its handoff. The repo reporter imports the same predicate, so it cannot
disagree and place the same legacy/contradictory snapshot in its ready bucket.
Fixtures probe each side of the gate, and the dedicated workflow owns their CI
enrollment.

## Intentional

- Legacy local snapshots that lack the versioned proof intentionally fail
  closed to attention. This temporarily disables automatic ready reporting
  until the local producer follow-up emits the contract; it preserves review
  and remediation wakes without creating false merge permission.
- The proof requires at least one required check. Atlas has required branch
  checks; an empty set is treated as missing enforcement, not green.
- This slice does not overload `live-reconciliation`, whose bot-accounting
  semantics deliberately differ from the strict all-thread merge invariant.

## Deferred

- `PR-Watcher-Live-Readiness-Producer`: move or regenerate the machine-local
  watcher from repository-owned source, fetch required checks and every
  review-thread page live, and emit the version-1 proof. Until that lands, the
  new consumers remain safely attention-only.
- Enabling GitHub's server-side required conversation-resolution setting is an
  operator/repository-policy action after current parallel PR lanes are
  assessed; this slice does not alter global branch protection.

Parked hardening: none.

## Verification

- `python -m pytest tests/test_codex_wake_bridge.py tests/test_codex_issue_queue.py tests/test_install_codex_wake_bridge.py tests/test_report_pr_watcher_state.py tests/test_audit_pr_watcher_safety.py -q` - 98 passed.
- `python scripts/maturity_sweep.py scripts --tests-root tests --baseline tests/maturity_sweep/baseline_scripts.json --min-score 8 --sensitive-glob 'scripts/**'` - ratchet passed with no new brittleness above baseline.
- `python scripts/audit_pr_watcher_safety.py --repo-root . --repo-only` - passed; watcher remains unable to merge.
- `python scripts/audit_workflow_security_posture.py .github/workflows` - passed with pre-existing warnings only.
- `python scripts/install_codex_wake_bridge.py --bin-dir <tmp>/bin --systemd-dir <tmp>/systemd && python scripts/install_codex_wake_bridge.py --check --bin-dir <tmp>/bin --systemd-dir <tmp>/systemd && <tmp>/bin/atlas-codex-wake-bridge --help` - installed copy matched source and executed standalone.
- `git diff --check` - passed.
- `python scripts/sync_pr_plan.py plans/PR-Watcher-Zero-Thread-Readiness.md --check` - passed.
- Pending before push: the single `push_pr.sh` local-review run.

## Estimated diff size

| File | LOC |
|---|---:|
| `.github/workflows/codex_wake_bridge_checks.yml` | 3 |
| `docs/long_running_session_watcher_handoff.md` | 38 |
| `plans/PR-Watcher-Zero-Thread-Readiness.md` | 137 |
| `scripts/codex_wake_bridge.py` | 112 |
| `scripts/report_pr_watcher_state.py` | 81 |
| `tests/test_codex_wake_bridge.py` | 169 |
| `tests/test_report_pr_watcher_state.py` | 84 |
| **Total** | **624** |

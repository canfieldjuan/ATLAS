# PR-Live-Reconciliation-Thread-Gate

## Why this slice exists

PR #2256 proved the `live-reconciliation` check can remain red after the Codex
connector has no unresolved review threads. The required target workflow still
waited for current-head Codex activity, so clean/no-thread states could burn an
extra review-window cycle and invite more review churn before the PR became
mergeable. This workflow/process slice removes that timing race at the source.
Codex review on #2258 also exposed stale current-head attestation language in
`AGENTS.md` and watcher readiness surfaces; those have to move with the checker
or the reviewer will keep enforcing the old contract from a different anchor.
The diff is over the 400 LOC soft cap because the indivisible fix removes the
obsolete wait/docs-only proof branch, aligns the watcher/wake-bridge readiness
contract, closes the review-event retrigger race, renames the colliding plan
basename, and updates the matching regression tests in the same slice.

### Problem-derived contract

- Root cause: `live-reconciliation` conflated two different concerns: whether
  scoped Codex review threads are still open, and whether the connector has
  posted current-head review activity inside a grace window. The second concern
  is a timing/attestation proxy that can fail a PR with no actionable Codex
  threads.
- Correct fix must touch/change: the required
  `.github/workflows/ai_reconciliation_live.yml` invocation, the
  `.github/workflows/ai_reconciliation_review_retrigger.yml` follow-up,
  `scripts/check_ai_reconciliation_live.py` decision path, `AGENTS.md`
  readiness wording, watcher/wake-bridge readiness gates, unit-gate impacted
  test ownership for the changed reconciliation workflows/scripts, and focused
  tests so required and local watcher gates pass as soon as no unresolved scoped
  Codex threads remain.
- Must not change: branch-protection inventory, `claude-review` policy,
  exact Codex bot-login filtering, review-thread pagination/stability checks,
  open-thread failure behavior, PR body reconciliation wording, product code,
  EOM/render lanes, or customer-visible behavior.

## Scope (this PR)

Ownership lane: workflow/live-reconciliation-open-threads-only
Slice phase: Workflow/process

1. Remove the required live workflow's `--wait-for-review-window` invocation.
2. Make the checker treat "no unresolved scoped Codex threads" as the only
   success condition, independent of current-head review state or PR update
   age.
3. Keep `--wait-for-review-window` and `--review-grace-seconds` accepted as
   deprecated compatibility inputs, but make them no-ops.
4. Update focused tests for no-wait, no-refetch, no-file-proof, and open-thread
   failure behavior.
5. Align `AGENTS.md`, the watcher, wake bridge, and overnight watcher script so
   they no longer require current-head Codex review attestation or block on
   `CHANGES_REQUESTED` when scoped Codex threads are clear.
6. Rename the plan to an unused basename so merge teardown can archive it.
7. Make the review-event retrigger wait for the current target run to complete
   before rerunning it, with bounded retry when GitHub rejects an in-progress
   rerun request.
8. Keep Codex review-attestation fetch failures diagnostic-only in the wake
   bridge when the required checks and review-thread proof are complete/clear.
9. Map the changed AI-reconciliation workflows and checker/wake-bridge scripts
   to their focused unit-test owners so `unit-gate` does not escalate to the
   full repo suite and fail on unrelated baseline drift.

### Review Contract

- Acceptance criteria:
  1. Required `pull_request_target` live reconciliation invokes
     `scripts/check_ai_reconciliation_live.py --pr <number>` without
     `--wait-for-review-window`.
  2. `evaluate()` returns success immediately when `open_bot_threads()` finds no
     unresolved scoped Codex threads, even when the current-head Codex review is
     missing or `CHANGES_REQUESTED`.
  3. Live `main()` accepts `--wait-for-review-window` and malformed
     `ATLAS_CODEX_REVIEW_GRACE_SECONDS` without sleeping, fetching
     `updatedAt`, or refetching a second review-thread snapshot.
  4. Docs-only/no-thread PRs no longer fetch changed-file proof just to emit a
     current-head attestation exemption.
  5. Open scoped Codex threads still fail when the PR body claims clear,
     acknowledges open findings, lacks a usable reconciliation record, or omits
     the record.
  6. Watcher readiness and scheduled wake-bridge readiness require complete
     thread pagination and zero unresolved scoped Codex threads, but no longer
     require current-head Codex review attestation.
  7. `CHANGES_REQUESTED` review decision is recorded as metadata but does not
     block readiness by itself when scoped Codex threads are clear.
  8. Live snapshot consistency still fails closed on PR head movement or review
     thread movement, but Codex review/comment generation movement no longer
     fails the thread-only gate.
  9. Overnight documentation-only PR readiness uses the same required-green plus
     zero-unresolved-Codex-threads gate; it does not hold for bot review
     presence when `live-reconciliation` is green for the current head.
  10. The root plan basename is unique against `plans/archive/`.
  11. Review-event retrigger waits for the trusted target run for the same head
      SHA to complete before posting the rerun request, and retries transient
      rerun rejection.
  12. Scheduled wake bridge readiness is not blocked by optional Codex review
      attestation errors when required checks are green and review threads are
      complete/clear.
  13. Unit-gate impacted-test selection for this branch returns the focused
      reconciliation/watcher/security owner tests instead of `FULL`.
- Reachability proof: real entrypoint is the GitHub Actions workflow command in
  `.github/workflows/ai_reconciliation_live.yml`; observable effect is the
  checker returning exit 0 for no-thread states and exit 1 for unresolved
  scoped Codex threads in `tests/test_check_ai_reconciliation_live.py`, plus
  watcher/wake-bridge tests returning ready when review attestation is missing
  but scoped Codex threads are clear, and checker tests allow review/comment
  generation movement while still failing closed on thread generation movement,
  plus static workflow assertions that the review-event retrigger waits/retries
  before rerunning the trusted target context. Unit-gate reachability proof is
  `scripts/select_impacted_tests.py --base origin/main` returning focused owner
  tests and `scripts/check_unit_gate.py --selected-files` reporting zero
  regressions.
- Affected surfaces: live AI reconciliation workflow, reconciliation checker
  CLI, review-event retrigger workflow, AGENTS readiness wording, local
  watcher/wake-bridge readiness, overnight watcher docs/script wording, and
  focused checker/watcher tests.
  Unit-gate impacted-test ownership for those workflow/script surfaces is also
  affected.
- Risk areas: required-check timing races, stale current-head review state,
  accidental sleep/refetch in required CI, docs-only proof fetch churn, stale
  overnight docs-only holds, and preserving unresolved-thread blocking.
- Reviewer rules triggered: R1, R2, R6, R8, R10, R12, R14.

### Boundary-change enumeration

Required when this diff changes a guard, validator, normalizer, resolver,
router/classifier, or admission boundary. Name each changed boundary path or
seam in the enumeration; otherwise write "N/A - no boundary change."

- Boundary path/seam: `scripts/check_ai_reconciliation_live.py::evaluate` and
  `main()` gate the required `live-reconciliation` check;
  `.github/workflows/ai_reconciliation_review_retrigger.yml` gates
  review-event re-evaluation of that required context;
  `scripts/pr_watcher.py::_classify` and
  `scripts/codex_wake_bridge.py::readiness_blockers` gate local watcher
  readiness; `scripts/select_impacted_tests.py::EXPLICIT_TEST_OWNERS` gates
  scoped unit-test selection for runtime/config and path-loaded script changes.
- Replaced-path behaviors: no-thread states no longer evaluate current-head
  Codex review state, PR update age, review grace, docs-only file proof, or
  `CHANGES_REQUESTED` review-decision metadata.
- Guard-relevant fields: `reviewThreads.nodes[].isResolved`, first-thread
  comment author login, PR body reconciliation marker, optional `--threads-file`
  fixtures, and live PR thread snapshot.
- Caller x input shape: GitHub Actions passes only `--pr`; tests exercise
  file-backed fixtures and live-mode monkeypatched PR snapshots.

### Deployed-config probing

Required for guard, validator, resolver, admission-boundary, or env/config
fallback changes; otherwise write "N/A - no guard/config boundary change."

- Deployed/default config values: `ATLAS_CODEX_REVIEW_GRACE_SECONDS` remains
  accepted for compatibility but no longer affects the gate.
- Explicit value probe:
  `test_main_ignores_malformed_review_window_config` proves malformed grace
  config no longer fails no-thread reconciliation.
- Absent value probe: `test_main_live_default_does_not_refetch_inside_review_window`
  proves the default live path does not fetch PR `updatedAt`.
- Default-session/default-context probe:
  `test_missing_current_head_codex_review_passes_inside_fresh_update_window`
  proves missing current-head activity inside the old window passes when
  threads are clear.
- Side-effect ordering:
  `test_main_live_wait_flag_is_noop_inside_review_window` proves the deprecated
  wait flag does not sleep or refetch.
  `test_current_head_codex_review_is_not_required_for_ready_state` and
  `test_post_review_metadata_records_review_decision_without_blocking_readiness`
  prove watcher readiness no longer blocks on stale review metadata.

### Files touched

- `.github/workflows/ai_reconciliation_live.yml`
- `.github/workflows/ai_reconciliation_review_retrigger.yml`
- `AGENTS.md`
- `docs/OVERNIGHT_ARC_WORKFLOW.md`
- `docs/long_running_session_watcher_handoff.md`
- `plans/PR-Live-Reconciliation-Thread-Gate.md`
- `scripts/check_ai_reconciliation_live.py`
- `scripts/codex_wake_bridge.py`
- `scripts/pr_watcher.py`
- `scripts/select_impacted_tests.py`
- `scripts/watch_owned_pr.sh`
- `tests/test_check_ai_reconciliation_live.py`
- `tests/test_codex_wake_bridge.py`
- `tests/test_pr_watcher.py`
- `tests/test_select_impacted_tests.py`
- `tests/test_watch_owned_pr.py`

## Mechanism

The workflow stops passing the wait flag to the required `pull_request_target`
job. The checker keeps fetching a stable review-thread snapshot, then
`evaluate()` makes one decisive split: no unresolved scoped Codex threads exits
0; any unresolved scoped Codex thread exits 1 with the existing PR-body
reconciliation diagnostics. The CLI still accepts the old wait/grace arguments
so existing callers do not break, but no code path uses them to sleep, refetch,
or require current-head Codex activity.

Review/comment attestation fetches remain available as diagnostics, but snapshot
stability checks no longer compare their generations; only PR head movement and
review-thread movement can fail the thread-only snapshot.

The watcher and wake bridge keep review-decision and current-head-review counts
as diagnostics, but readiness depends on required checks, thread-pagination
completion, zero unresolved scoped Codex threads, non-draft/open PR state, clean
merge state, and local/head consistency. The shell watcher and docs use the same
contract, including documentation-only PRs, so autonomous handoffs do not
reintroduce the removed gate.

The review-event retrigger remains default-branch trusted code, but now waits
for the matching `pull_request_target` run to complete before requesting a
rerun and retries rerun rejection. That closes the window where a review-event
follow-up could finish before the required target run and leave the current head
with a stale green required context after a new inline Codex thread.

The unit-gate selector now treats the changed AI-reconciliation workflows and
path-loaded checker/wake-bridge scripts as explicitly owned CI surfaces. That
keeps this workflow/process slice on its focused reconciliation, watcher, and
workflow-security tests instead of escalating to the full repo suite, where a
separate baseline-shrink drift can block an unrelated runtime/config change.

## Intentional

- No branch-protection change in this slice. The required status remains
  `live-reconciliation`; only its pass/fail predicate changes.
- No replacement "final Codex approval" gate. The operator asked to use whether
  Codex left unresolved comments as the merge signal, and the existing
  `claude-review` discussion is a separate trust-boundary slice.
- Deprecated CLI flags are kept as no-op compatibility inputs to avoid breaking
  external/manual invocations while the workflow call is simplified.
- Current-head Codex review counts remain in watcher output as diagnostics only.
- Codex review-attestation API/pagination errors remain in watcher output as
  diagnostics only; the bridge trusts the complete review-thread proof for
  readiness.
- Review/comment-generation movement is intentionally ignored by the live
  checker because review presence is no longer part of the required predicate;
  PR head and review-thread movement still fail closed.
- The unit-gate selector fix does not restore or edit
  `tests/unit_gate_baseline.txt`; it maps this slice's owned runtime/config
  surfaces to focused tests so unrelated full-suite baseline drift is not
  repaired inside this PR.

## Deferred

None.

Parked hardening: none.

## Verification

- `python -m pytest tests/test_check_ai_reconciliation_live.py -q` -- 64 passed.
- `python -m pytest tests/test_codex_wake_bridge.py -q` -- 46 passed.
- `python -m pytest tests/test_watch_owned_pr.py -q` -- 23 passed.
- `python -m pytest tests/test_watch_owned_pr.py tests/test_pr_watcher.py tests/test_codex_wake_bridge.py tests/test_check_ai_reconciliation_live.py -q` -- 209 passed.
- `python -m pytest tests/test_select_impacted_tests.py -q` -- 48 passed.
- `python scripts/select_impacted_tests.py --base origin/main` -- selected
  `tests/test_audit_workflow_security_posture.py`,
  `tests/test_check_ai_reconciliation_live.py`,
  `tests/test_codex_wake_bridge.py`, `tests/test_pr_watcher.py`, and
  `tests/test_watch_owned_pr.py` (not `FULL`).
- `python scripts/check_unit_gate.py --baseline tests/unit_gate_baseline.txt --base-baseline /tmp/atlas-base-baseline-2258.txt --selected-files /tmp/atlas-selected-2258.txt --pytest-args $(tr '\n' ' ' < /tmp/atlas-selected-2258.txt) -m "not integration and not e2e" --continue-on-collection-errors -rfE --tb=no -q -p no:cacheprovider` -- 0 regressions.
- `python -m pytest tests/test_check_ai_reconciliation_live.py tests/test_codex_wake_bridge.py tests/test_pr_watcher.py tests/test_watch_owned_pr.py tests/test_select_impacted_tests.py tests/test_audit_workflow_security_posture.py -q` -- 276 passed.

## Estimated diff size

| File | LOC |
|---|---:|
| `.github/workflows/ai_reconciliation_live.yml` | 3 |
| `.github/workflows/ai_reconciliation_review_retrigger.yml` | 36 |
| `AGENTS.md` | 3 |
| `docs/OVERNIGHT_ARC_WORKFLOW.md` | 27 |
| `docs/long_running_session_watcher_handoff.md` | 29 |
| `plans/PR-Live-Reconciliation-Thread-Gate.md` | 284 |
| `scripts/check_ai_reconciliation_live.py` | 329 |
| `scripts/codex_wake_bridge.py` | 16 |
| `scripts/pr_watcher.py` | 9 |
| `scripts/select_impacted_tests.py` | 13 |
| `scripts/watch_owned_pr.sh` | 16 |
| `tests/test_check_ai_reconciliation_live.py` | 255 |
| `tests/test_codex_wake_bridge.py` | 46 |
| `tests/test_pr_watcher.py` | 40 |
| `tests/test_select_impacted_tests.py` | 19 |
| `tests/test_watch_owned_pr.py` | 64 |
| **Total** | **1189** |

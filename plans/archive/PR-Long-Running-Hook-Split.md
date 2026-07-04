# PR-Long-Running-Hook-Split

## Why this slice exists

Issue #1962 S2 documented the first long-running watcher runbook, and S3
measured CI runtime/duplication. The live #1968 fix loop exposed the next
operating-model gap: the docs still describe one generic "poll every 30
minutes" watcher, while the actual safe loop needs to distinguish review-event
attention from scheduled green confirmation.

The root cause is that review-event attention and CI-green merge confirmation
are different signals with different safety semantics, but the current runbook
collapses them into one polling story. This slice fixes the root at the docs and
operating-contract layer: it names review-event attention as non-autonomous
unless an operator or external bridge wakes the session, preserves the scheduled
green-confirmation hook as the autonomous path, updates AGENTS.md with the
recorded-gap fallback for environments without a concrete event bridge, and
documents the standing-authorization merge rule plus head-mismatch recovery
rule. It does not change workflow code or local watcher implementation.

## Scope (this PR)

Ownership lane: workflow/autonomous-ci-cd-map
Slice phase: Workflow/process

1. Update the long-running watcher handoff to distinguish review-event
   attention from scheduled CI-green confirmation.
2. Update the CI/CD map so the operating loop distinguishes event-driven review
   signals from scheduled merge confirmation.
3. Align AGENTS.md with the handoff: if no concrete push/review-event hook is
   available, record the gap and rely on the scheduled watcher as the autonomous
   fallback.
4. Document that an explicit standing authorization from the operator lets the
   builder merge after a scheduled `ready_for_human_merge` wake-up, but only
   after the normal AGENTS ownership/head/check/thread guards pass.
5. Document the #1968 head-mismatch recovery rule: stop on unexpected remote
   head movement, fetch/inspect the delta, and do not force-push over it.

### Review Contract

Acceptance criteria:
- The runbook names review-event attention and scheduled green confirmation,
  and states that only the scheduled watcher is an autonomous local wake-up
  today.
- The runbook and AGENTS.md both state how to record an unavailable
  push/review-event hook without claiming immediate wake-up coverage.
- The scheduled green-confirmation path preserves `AUTO_MERGE=0`; the builder
  performs the merge only when explicitly authorized for the arc.
- The canonical AGENTS scheduled merge gate includes clean merge-conflict /
  mergeability state, not only green checks and clean review/reconciliation.
- The review-event path is described as an attention/fix trigger, not a merge
  trigger, and does not claim to wake the builder without an operator or
  explicit bridge.
- The canonical AGENTS rule and handoff both reserve standing-authorized merge
  for scheduled green-confirmation wakes; push/review-event wakes can only
  inspect, fix, or record readiness.
- The session-state template has durable fields for the push/review-event hook,
  timer hook, next timer wake, and last watcher state.
- The session-state template has a durable standing-authorization field so merge
  permission does not depend on memory after compaction.
- The session-state template allowed-action enum includes the scheduled-ready
  guarded merge mode.
- The event-bridge contract requires a recorded event hook to wake the builder
  and does not tell sessions to run the scheduled watcher command as a
  review-event bridge that could grant merge permission.
- The docs explicitly forbid active GitHub polling loops between watcher
  wake-ups.
- The docs include the head-mismatch stop/fetch/inspect rule learned from
  #1968.
- The standing-authorization merge guard list includes review-thread status and
  merge-conflict/mergeability state.
- The PR is documentation/plan only; it must not change GitHub Actions, local
  watcher scripts, branch protection, or CI behavior.

Affected surfaces:
- Developer workflow docs for long-running Atlas builder sessions.

Risk areas:
- Accidentally implying a merge-capable watcher.
- Blurring "review changed" with "ready to merge."
- Encouraging force-push over another actor's remote head.
- Weakening AGENTS.md by hiding the missing event-hook bridge instead of making
  the gap explicit.

Triggered reviewer rules:
- R1 Requirements match
- R2 Test evidence
- R6 Workflow/process
- R14 Codebase verification

### Files touched

- `AGENTS.md`
- `docs/SESSION_STATE_TEMPLATE.md`
- `docs/ci_cd_autonomous_coding_map.md`
- `docs/long_running_session_watcher_handoff.md`
- `plans/PR-Long-Running-Hook-Split.md`

## Mechanism

The runbook keeps the watcher read-only and `AUTO_MERGE=0`, but gives the
builder a precise operating contract:

```text
push/review-event attention -> inspect/fix owned PR only when operator/bridge wakes
                              record readiness if green, but never merge
30-minute watcher -> if ready_for_human_merge and standing authorization exists,
                     run AGENTS merge guards, merge, teardown, continue
```

The CI/CD map mirrors that contract in the high-level operating loop so future
sessions do not treat "polling" as active GitHub sampling.

The session-state template gets the same hook/timer fields so a clean-start
session has a durable slot for whether immediate review-event wake-up coverage
exists and whether standing merge authorization was explicitly granted for the
arc.

## Intentional

- No workflow, branch-protection, or watcher executable changes. This slice
  makes the operating rule durable before any implementation changes.
- The push/review-event path is not documented as autonomous unless an external
  bridge exists. A local webhook bridge is deferred until the monitoring slice
  proves it is needed.
- `AUTO_MERGE=0` remains mandatory. The builder can merge after a scheduled
  green confirmation only because the operator gave explicit standing
  authorization for this arc.

## Deferred

- #1962 S5 monitoring/reporting spec: decide whether to add a local
  review-event notification bridge beyond GitHub Actions and desktop
  notifications.
- Applying CI speedup candidates from `docs/ci_cd_runtime_duplication_audit.md`.

Parked hardening: none.

## Verification

- `python scripts/sync_pr_plan.py plans/PR-Long-Running-Hook-Split.md --check` - passed.
- `git diff --check` - passed.
- `bash scripts/local_pr_review.sh --current-pr-body-file /tmp/long_running_hook_split_pr_body.md` - passed; non-blocking warning only: plans archive backlog exceeds threshold.

## Estimated diff size

| File | LOC |
|---|---:|
| `AGENTS.md` | 18 |
| `docs/SESSION_STATE_TEMPLATE.md` | 13 |
| `docs/ci_cd_autonomous_coding_map.md` | 24 |
| `docs/long_running_session_watcher_handoff.md` | 87 |
| `plans/PR-Long-Running-Hook-Split.md` | 154 |
| **Total** | **296** |

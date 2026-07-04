# PR-Codex-Wake-Bridge

## Why this slice exists

Issue #1962 now has a concrete long-running-session gap: the local watcher can
write `ready_for_human_merge` or `review_changed` to JSON/log/desktop
notifications, but that does not wake a Codex session by itself. The operator
asked to build that bridge before continuing the arc.

Root cause: watcher state and agent resume input live in separate, unconnected
surfaces. `atlas-pr-watch` records the PR state, but no repo-owned command turns
that snapshot into a guarded Codex resume/check prompt or a deterministic handoff
artifact. This PR fixes that bridge layer. It does not make the watcher itself
merge-capable.

This slice is over the 400 LOC soft cap because the bridge's safety boundary,
operator docs, and failure-mode tests have to land together. Splitting the
script from its no-merge/pending/malformed-state tests would ship the exact kind
of local automation surface this lane is trying to make safer.

## Scope (this PR)

Ownership lane: ci-cd/autonomous-codex-wake-bridge
Slice phase: Workflow/process

1. Add a repo-owned `scripts/codex_wake_bridge.py` that consumes a local watcher
   snapshot, classifies the wake source as scheduled-green, event-attention, or
   attention, and writes both JSON and Markdown handoff artifacts under the
   local watcher state directory.
2. Support an opt-in `--run-command` path, or a `CODEX_WAKE_COMMAND` entry in
   the per-watcher config file, that can invoke an operator-configured Codex
   command with the prompt on stdin, while defaulting to write-only dry handoff
   behavior.
3. Update the watcher handoff and CI/CD map docs so future sessions know the
   distinction between watcher state, wake bridge handoff, and guarded merge
   authority.
4. Add focused unit tests for state classification, prompt content, command
   invocation, and the no-merge boundary.
5. Add a PR-head CI workflow for the wake bridge tests so bridge behavior is
   exercised from the pull request code, while the existing `pull_request_target`
   pre-push audit remains trusted-base only.

### Review Contract

- [ ] `scripts/codex_wake_bridge.py` never calls `gh pr merge`, never edits a PR,
      and never polls GitHub; it consumes existing watcher JSON/config/state only.
- [ ] Scheduled `ready_for_human_merge` produces a prompt that tells Codex to run
      the AGENTS merge guards before any merge, and preserves that merge is
      scheduled-ready-only.
- [ ] Push/review-event or review-change wakes produce attention-only prompts
      that forbid merge and direct the session to inspect/fix the owned PR.
- [ ] Scheduled wakes treat pending check details as non-ready even if the
      watcher state is stale or contradictory.
- [ ] The optional command runner is explicit opt-in and receives the generated
      prompt on stdin rather than interpolating untrusted prompt text into a
      shell command.
- [ ] Tests cover malformed or non-ready watcher states so the bridge fails
      closed to handoff-only behavior.
- [ ] CI runs the bridge unit tests against PR-head code in a read-only
      `pull_request` workflow; trusted-base `pull_request_target` jobs continue
      to inspect PR code only as data.

### Files touched

- `.github/workflows/codex_wake_bridge_checks.yml`
- `.github/workflows/pre_push_audit.yml`
- `docs/ci_cd_autonomous_coding_map.md`
- `docs/long_running_session_watcher_handoff.md`
- `plans/PR-Codex-Wake-Bridge.md`
- `scripts/codex_wake_bridge.py`
- `tests/test_codex_wake_bridge.py`

## Mechanism

The bridge reads:

```text
~/.config/atlas-pr-watchers/<session-id>.env
~/.local/state/atlas-pr-watchers/<session-id>.json
SESSION_STATE.local.md (path from watcher config, if present)
```

It derives a narrow `wake_kind`:

- `scheduled-ready` only when the watcher state is `ready_for_human_merge` and
  the caller says `--source scheduled`.
- `event-attention` for actionable `--source event` wakes, because review or
  push events are not merge signals but should still wake a session even while
  scheduled CI is pending.
- `pending` for scheduled pending watcher states or non-empty pending-check
  lists, which writes a handoff but does not run the optional command by default.
- `attention` for red/review-changed/head-mismatch/dirty states.
- `invalid-snapshot` for missing or malformed watcher JSON, which writes a
  handoff but never runs the optional command.

It writes `.wake.json` and `.wake.md` next to the watcher JSON. The Markdown is
the resumable prompt for `codex exec -C <repo> -` or an interactive paste. When
`--run-command` is supplied, the command is parsed with `shlex.split` and run
with the generated prompt as stdin; the prompt itself is never shell-expanded.
The command runner also refuses stale or missing `REPO_DIR` values instead of
launching from an unrelated current directory.

## Intentional

- The slice does not modify the local `~/.local/bin/atlas-pr-watch` script. That
  file is local infrastructure, not repo source; this PR adds the repo-owned
  bridge command and docs for how the local watcher should call it.
- The default behavior does not launch Codex. Write-only handoff is safe on any
  machine; actual agent launching is explicitly configured by the operator.
- The bridge is not a GitHub webhook receiver. Event delivery remains an
  operator/environment concern; this slice defines the command the event bridge
  should call.

## Deferred

- Installing machine-local systemd/webhook units that call the bridge is
  deferred because those files live outside the repository and vary by machine.
- A hosted GitHub App/webhook receiver that directly wakes Codex is deferred to a
  future infrastructure slice if local hooks are not enough.

Parked hardening: none.

## Verification

- `python -m pytest tests/test_codex_wake_bridge.py -q` -- 13 passed.
- `python -m py_compile scripts/codex_wake_bridge.py tests/test_codex_wake_bridge.py` -- passed.
- `python scripts/maturity_sweep_file_lane.py scripts/codex_wake_bridge.py --tests-root tests --json` -- score 3.
- `python scripts/audit_workflow_security_posture.py .github/workflows` -- passed (pre-existing SHA-pin warnings only).
- `git diff --check` -- passed.
- `bash -lc '! rg -n "gh pr (merge|edit|checks|view|comment)|gh api|shell=True" scripts/codex_wake_bridge.py'` -- passed, no forbidden command path.
- `python scripts/sync_pr_plan.py plans/PR-Codex-Wake-Bridge.md --check` -- passed.
- `ATLAS_CURRENT_PR_BODY_FILE=/tmp/codex_wake_bridge_pr_body.md bash scripts/local_pr_review.sh` -- passed.

## Estimated diff size

| File | LOC |
|---|---:|
| `.github/workflows/codex_wake_bridge_checks.yml` | 29 |
| `.github/workflows/pre_push_audit.yml` | 4 |
| `docs/ci_cd_autonomous_coding_map.md` | 3 |
| `docs/long_running_session_watcher_handoff.md` | 133 |
| `plans/PR-Codex-Wake-Bridge.md` | 145 |
| `scripts/codex_wake_bridge.py` | 398 |
| `tests/test_codex_wake_bridge.py` | 413 |
| **Total** | **1125** |

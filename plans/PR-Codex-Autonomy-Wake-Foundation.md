# PR-Codex-Autonomy-Wake-Foundation

## Why this slice exists

Issue #1962 is now a living tracker for the autonomous CI/CD operating model.
The current documented Codex wake bridge can write a handoff and can run an
operator-provided command, but the local systemd service installed on this
machine still calls bare `atlas-pr-watch` instead of the documented
`atlas-pr-watch-and-wake` wrapper. That means the watcher records state but does
not wake Codex. The same local state also showed a stale timer repeatedly
failing against a deleted worktree.

Root cause: the bridge contract was documented but not made easy to install or
verify as repo-owned tooling, and event-source wakes were too broad to be a safe
autonomous trigger. This slice fixes the root for the first autonomous step:
make the scheduled watcher capable of launching Codex through a checked wrapper,
make review/event wakes attention-only, and add tests so the setup cannot drift
back to "state-only" without being noticed.

Diff budget note: this is over the 400 LOC soft cap because the behavior, its
installer, its source-aware wake tests, and the safety documentation need to land
together. Splitting the tests or docs out would recreate the exact
"documented-but-not-installed" gap this slice fixes.

## Scope (this PR)

Ownership lane: workflow/codex-autonomy
Slice phase: Workflow/process

1. Add repo tooling that installs/verifies the local
   `atlas-pr-watch-and-wake` wrapper, trusted bridge copy, and systemd drop-in
   so the timer runs the watcher and then the installed Codex wake bridge.
2. Tighten `scripts/codex_wake_bridge.py` event-source classification so event
   wakes remain attention-only, while scheduled wakes remain the only
   merge-consideration path.
3. Document the autonomous decision rule: durable technical fix by default;
   operator-owned decisions defer to GitHub/email in a later slice instead of
   becoming chat-stop menus.
4. Keep this PR below the queue/email continuation boundary.

### Review Contract

Acceptance criteria:

- A reviewer can run the installer in `--check` mode and see whether the local
  systemd service calls `atlas-pr-watch-and-wake`.
- The generated wrapper runs `atlas-pr-watch`, then
  the installed `atlas-codex-wake-bridge --source scheduled` copy rather than
  executing bridge code out of the watched PR worktree, and never contains merge
  commands.
- Event-source bridge wakes are actionable inspection/fix handoffs for pending,
  ready, review-changed, and failure snapshots, but never grant merge authority.
- Scheduled `ready_for_human_merge` keeps producing a guarded merge prompt, but
  the prompt still requires explicit standing authorization and live AGENTS
  guards.
- The docs name the "technical fork is not a menu" rule and defer issue-queue
  plus email notification to the next slice.

Affected surfaces: local Codex watcher setup, wake-bridge classification,
long-running-session docs, session template.

Risk areas: accidentally granting merge authority to watcher/systemd, making
review-event wakes unable to respond to real feedback, or breaking the existing
manual handoff flow.

Triggered reviewer rules: R2 test evidence, R14 codebase verification.

Reachability proof: run the new installer in `--check` mode against fixture
directories and run `tests/test_codex_wake_bridge.py` to prove event/scheduled
wake behavior.

### Files touched

- `.github/workflows/codex_wake_bridge_checks.yml`
- `docs/SESSION_STATE_TEMPLATE.md`
- `docs/autonomous_coding_repo_playbook.md`
- `docs/long_running_session_watcher_handoff.md`
- `plans/PR-Codex-Autonomy-Wake-Foundation.md`
- `scripts/codex_wake_bridge.py`
- `scripts/install_codex_wake_bridge.py`
- `tests/test_codex_wake_bridge.py`
- `tests/test_install_codex_wake_bridge.py`

## Mechanism

- Add `scripts/install_codex_wake_bridge.py` as a small local-setup tool. In
  normal mode it writes the wrapper, trusted bridge copy, and systemd drop-in;
  in `--check` mode it validates the current local files without mutation. Tests
  use temporary `$HOME`-style directories and never touch real systemd.
- Update `classify_wake()` so `source=event` preserves immediate
  push/review-event wakes as attention-only for pending, ready, and review
  states while still forbidding merge from event wakes.
- Update the handoff docs and session template so future Codex sessions record
  issue queue / operator email fields, but this slice does not implement queue
  selection or email sending yet.

## Intentional

- No watcher merge path is added. The watcher and wrapper remain read-only with
  respect to GitHub writes.
- The installer is local tooling, not a GitHub Action. Autonomy depends on the
  operator's workstation and must be explicit per machine/session.
- Issue-queue continuation and defer/email notification are intentionally left
  to the next slice so this PR can first make waking Codex reliable.

## Deferred

- Next slice: issue-queue continuation plus defer-and-email notification for
  operator-owned decisions.
- Later slice: optional push/review webhook bridge. This PR only makes the
  scheduled local systemd path and event-source bridge semantics safe.

Parked hardening: none.

## Verification

- `python -m pytest tests/test_codex_wake_bridge.py tests/test_install_codex_wake_bridge.py tests/test_audit_pr_watcher_safety.py -q` - 43 passed.
- `python scripts/maturity_sweep.py scripts --tests-root tests --baseline tests/maturity_sweep/baseline_scripts.json --min-score 8 --sensitive-glob 'scripts/**'` - ratchet gate passed.
- `rm -rf /tmp/atlas-watch-bin /tmp/atlas-watch-systemd && python scripts/install_codex_wake_bridge.py --bin-dir /tmp/atlas-watch-bin --systemd-dir /tmp/atlas-watch-systemd && python scripts/install_codex_wake_bridge.py --check --bin-dir /tmp/atlas-watch-bin --systemd-dir /tmp/atlas-watch-systemd` - install/check smoke passed.
- `ATLAS_SESSION_STATE_FILE=SESSION_STATE.codex-autonomy-wake-foundation-1962.local.md bash scripts/local_pr_review.sh --current-pr-body-file /tmp/pr-body-codex-autonomy-wake-foundation.md` - local PR review passed.

## Estimated diff size

| File | LOC |
|---|---:|
| `.github/workflows/codex_wake_bridge_checks.yml` | 10 |
| `docs/SESSION_STATE_TEMPLATE.md` | 5 |
| `docs/autonomous_coding_repo_playbook.md` | 4 |
| `docs/long_running_session_watcher_handoff.md` | 105 |
| `plans/PR-Codex-Autonomy-Wake-Foundation.md` | 135 |
| `scripts/codex_wake_bridge.py` | 13 |
| `scripts/install_codex_wake_bridge.py` | 174 |
| `tests/test_codex_wake_bridge.py` | 67 |
| `tests/test_install_codex_wake_bridge.py` | 217 |
| **Total** | **730** |

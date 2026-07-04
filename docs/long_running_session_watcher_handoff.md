# Long-Running Session Watcher Handoff

Issue: #1962

Use this handoff when the operator wants a builder session to keep an owned PR
moving while they are away. It complements `AGENTS.md` section 3c.1 and the CI
map in `docs/ci_cd_autonomous_coding_map.md`.

Important mode split:

- **Claude Code native sessions** should use Claude Code's PR subscription,
  review reactivity, and 30-minute polling. Do not force those sessions onto
  the local systemd `atlas-pr-watch` timer unless the operator explicitly asks
  for local state files too.
- **Codex/local CLI sessions** need a separate wake bridge for true autonomous
  resume. `atlas-pr-watch` can write watcher JSON/log state, but it cannot wake
  Codex by itself. A bridge must start or resume a Codex run with the watcher
  state and a prompt to read this session's state file, rerun guards, and act
  only on the owned PR.
- **Both modes** keep merge authority with the active builder only. A watcher,
  timer, notification, or bridge can report state; it cannot merge.

## What Changed

Long-running sessions now have two durable responsibilities:

1. Keep this session's state file current for the owned lane and PR.
2. Record the actual wake mode after each PR open or push: Claude Code native
   subscription/polling, a Codex wake bridge, or local watcher state-only.

The local watcher is intentionally local and per session. One Codex/local
session equals one watcher config. A second Codex/local session should create a
second config and timer rather than reuse another session's watcher. Claude Code
native sessions do not need this local watcher path.

## Wake Modes

Long-running sessions should not actively poll GitHub in an in-chat loop. They
wait for the session's wake mode, then take the narrow action that signal
allows.

| Mode | Source | Builder action | Autonomous today? | Merge allowed? |
|---|---|---|---|---|
| Claude Code native | Claude Code PR subscription/review reactivity plus 30-minute polling | Claude Code resumes as the active builder, inspects only the owned PR, fixes actionable feedback, and runs merge guards when scheduled polling reports ready | Yes, when Claude Code subscription is active | Active builder only after explicit operator authorization and fresh guards |
| Codex wake bridge | External wrapper that starts/resumes Codex with watcher state | Fresh/active Codex reads this session's state file, runs `scripts/report_pr_watcher_state.py`, then fixes, waits, reports ready, or runs guarded merge if authorized | Only when the bridge exists | Active builder only after explicit operator authorization and fresh guards |
| Local watcher state-only | `atlas-pr-watch@<session>.timer` writes JSON/log state every 30 minutes | No agent wakes automatically; the next active agent consumes the state with `scripts/report_pr_watcher_state.py` | No | No |
| Operator signal | Human says "review is up", "green", or "merge" | Active builder inspects the owned PR and runs the same guards | Manual | Active builder only after explicit operator authorization and fresh guards |

The push/review-event path exists to reduce red-review latency when something
wakes the builder. It is an attention signal, not a merge signal. If no concrete
bridge exists for Codex/local sessions, record `Wake bridge: unavailable` in
this session's state file; the scheduled watcher remains only a state recorder
until an active agent consumes its output.

The scheduled poll/wake is the canonical readiness signal. A push/review-event
wake that observes a clean PR records readiness and waits for the scheduled
Claude poll or Codex wake-bridge confirmation before an active builder considers
a guarded merge.

## Local Watcher Files For Codex/Local State

These files are local machine infrastructure, not committed repo files:

| Path | Purpose |
|---|---|
| `~/.local/bin/atlas-pr-watch` | One-shot watcher executable |
| `~/.config/systemd/user/atlas-pr-watch@.service` | User systemd service template |
| `~/.config/systemd/user/atlas-pr-watch@.timer` | User systemd timer template, every 30 minutes |
| `~/.config/atlas-pr-watchers/<session-id>.env` | One config per builder session |
| `~/.local/state/atlas-pr-watchers/<session-id>.json` | Latest machine-readable watcher status |
| `~/.local/state/atlas-pr-watchers/<session-id>.wake.json` | Latest Codex wake-bridge handoff metadata |
| `~/.local/state/atlas-pr-watchers/<session-id>.wake.md` | Pasteable/resumable Codex wake prompt |
| `~/.local/state/atlas-pr-watchers/<session-id>.log` | Append-only watcher log |

If a Codex/local session is using local watcher state and
`~/.local/bin/atlas-pr-watch` is missing, stop and ask the operator. Do not
recreate a merge-capable watcher from scratch. The watcher must stay read-only
with respect to GitHub merges.

## Push/Review-Event Hook

AGENTS.md requires a real wake path for fully immediate long-running operation.
Claude Code native subscription satisfies that path for Claude Code sessions.
For Codex/local sessions, the hook is outside this repo unless the operator has
provided an integration. Use this concrete contract:

1. Record the hook in this session's state file as
   `Push/review-event hook: <name and trigger>`.
2. Configure the external bridge to start or resume the Codex/local builder
   session on new pushes, review threads, review events, and reconciliation
   events for the owned PR. Do not use the scheduled
   `~/.local/bin/atlas-pr-watch "${SESSION_ID}"` command as the event bridge
   unless it has a source-aware event mode that cannot grant merge permission.
3. If no such bridge exists, record `Wake bridge: unavailable`. The scheduled
   watcher can still record `review_changed`, but it is not autonomous until an
   active agent consumes the state. Operator-only notifications are a manual
   fallback, not a recorded push/review-event hook.

The unavailable state is safe but not autonomous. Do not describe that session
as having review-event wake-up coverage.

## Codex Wake Bridge

The watcher records state; it does not wake Codex by itself. Use
`scripts/codex_wake_bridge.py` to convert an existing watcher snapshot into a
resumable handoff:

```bash
python scripts/codex_wake_bridge.py "${SESSION_ID}" --source scheduled
```

The bridge reads the watcher config and JSON state, then writes:

```text
~/.local/state/atlas-pr-watchers/${SESSION_ID}.wake.json
~/.local/state/atlas-pr-watchers/${SESSION_ID}.wake.md
```

The Markdown file is the prompt for a resumed or `codex exec` run. By default
the bridge only writes handoff files. To launch a local command, pass
`--run-command` explicitly, or add a quoted `CODEX_WAKE_COMMAND` line to that
session's watcher config:

```bash
CODEX_WAKE_COMMAND="codex exec -C <repo-dir> -"
```

The command receives the generated prompt on stdin. The prompt text is not
interpolated into a shell command.

Wake-source rules:

- `--source event` is attention-only. It can wake Codex to inspect review
  activity or a new push, but it must not merge, even if the watcher snapshot is
  green.
- `--source scheduled` may classify `ready_for_human_merge` as
  `scheduled-ready`, which is only permission to run the AGENTS merge guards.
  The resumed builder still needs explicit standing merge authorization in
  this session's state file before merging.
- Pending states write a handoff but do not run the optional command by default.
  Do not replace the watcher timer with an in-chat polling loop.

This gives the two-hook shape without making the watcher merge-capable: event
hooks can call the bridge with `--source event`; the 30-minute green-confirmation
timer can call it with `--source scheduled`.

## Safety Rules

- No auto-merge. Truthy watcher auto-merge config is unsafe and must surface as
  attention, not an action path.
- A green PR becomes `ready_for_human_merge`; it does not merge itself.
- Local review runs `scripts/audit_pr_watcher_safety.py`; unsafe watcher configs
  or watcher source merge commands are blocking.
- Codex/local active builders must run `scripts/report_pr_watcher_state.py` on
  resume to consume ready, attention, pending, and stale watcher states.
  Desktop notifications are advisory only.
- Claude Code native sessions use Claude Code's PR subscription and 30-minute
  polling; local systemd watcher setup is optional, not required.
- If the operator gives explicit standing authorization for an arc, the active
  builder may merge after a scheduled `ready_for_human_merge` wake-up, but only
  after re-running the AGENTS pre-merge guards: open PR list, `origin/main`
  log, ownership guard, matching head SHA, current checks, review-thread
  status, live reconciliation, and merge-conflict/mergeability state.
- Standing authorization applies only to the scheduled green-confirmation wake,
  never to a push/review-event attention wake.
- Review/comment events do not currently wake a Codex/local builder session by
  themselves. Treat them as fast attention only when the operator or another
  explicit integration wakes the session; otherwise record the wake bridge as
  unavailable and use the scheduled watcher as a state recorder.
- The watcher may alert on red CI, pending CI, new review activity, head SHA
  mismatch, or failed AI reconciliation.
- A builder may fix only the owned PR and only the files allowed by the active
  slice or PR-fix-mode block.
- After merge, disable that PR's timer and tear down only that session's owned
  worktree/branch.
- If the watcher reports a head SHA mismatch, stop before any force-push or
  merge. Fetch the remote branch, inspect the unexpected delta, and either
  fast-forward and repair in a new commit or ask the operator if ownership is
  unclear.
- Between watcher wake-ups, do not run an ad hoc `gh` polling loop just to see
  whether CI is green.

This protects against the race where checks turn green before late comments or
review threads land.

## Setup For A Codex/Local Watcher Session

Skip this section for Claude Code native sessions unless the operator explicitly
asks for local watcher JSON/log state in addition to Claude's subscription.

Pick a stable session id:

```bash
SESSION_ID="<lane-slug>-<pr-number>"
SESSION_STATE_FILE="<absolute repo or worktree path>/SESSION_STATE.${SESSION_ID}.local.md"
export ATLAS_SESSION_STATE_FILE="${SESSION_STATE_FILE}"
```

Create the watcher config after the PR is opened or after an existing owned PR
is assigned to the session:

```bash
mkdir -p ~/.config/atlas-pr-watchers ~/.local/state/atlas-pr-watchers

cat > ~/.config/atlas-pr-watchers/${SESSION_ID}.env <<'EOF'
LABEL="<human label>"
REPO_DIR="<absolute repo or worktree path>"
PR="<pr number>"
REPO="canfieldjuan/ATLAS"
SESSION_STATE="<absolute path to SESSION_STATE.<session-id>.local.md>"
HEAD_SHA="<current PR head SHA>"
POLL_MINUTES="30"
AUTO_MERGE="0"
NOTIFY="1"
# Optional, quoted. Leave unset for write-only handoff.
# CODEX_WAKE_COMMAND="codex exec -C <absolute repo or worktree path> -"
EOF
```

Install the bridge wrapper and systemd drop-in. The wrapper reads `REPO_DIR`
from the session config each time it runs, so one systemd template can safely
serve multiple sessions without baking one worktree path into every timer.

```bash
cat > ~/.local/bin/atlas-pr-watch-and-wake <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
session_id="${1:?watcher session id required}"
config="${HOME}/.config/atlas-pr-watchers/${session_id}.env"

if [ ! -f "$config" ]; then
  echo "watcher config not found: $config" >&2
  exit 2
fi

# shellcheck disable=SC1090
source "$config"

if [ -z "${REPO_DIR:-}" ] || [ ! -d "$REPO_DIR" ]; then
  echo "invalid REPO_DIR for ${session_id}: ${REPO_DIR:-}" >&2
  exit 2
fi

~/.local/bin/atlas-pr-watch "${session_id}"
cd "$REPO_DIR"
python scripts/codex_wake_bridge.py "${session_id}" --source scheduled
EOF
chmod +x ~/.local/bin/atlas-pr-watch-and-wake

mkdir -p ~/.config/systemd/user/atlas-pr-watch@.service.d
cat > ~/.config/systemd/user/atlas-pr-watch@.service.d/wake-bridge.conf <<'EOF'
[Service]
ExecStart=
ExecStart=%h/.local/bin/atlas-pr-watch-and-wake %i
EOF
```

Run one manual poll through the same wrapper the timer will use:

```bash
~/.local/bin/atlas-pr-watch-and-wake "${SESSION_ID}"
```

Enable the 30-minute timer:

```bash
systemctl --user daemon-reload
systemctl --user enable --now "atlas-pr-watch@${SESSION_ID}.timer"
```

Check status:

```bash
systemctl --user list-timers 'atlas-pr-watch*'
journalctl --user -u "atlas-pr-watch@${SESSION_ID}.service" -n 80 --no-pager
cat ~/.local/state/atlas-pr-watchers/${SESSION_ID}.json
```

Disable after merge or reassignment:

```bash
systemctl --user disable --now "atlas-pr-watch@${SESSION_ID}.timer"
```

## Watcher States

| State | Meaning | Builder action |
|---|---|---|
| `pending` | At least one check is still pending | Record the next poll; do not ask the operator to babysit CI |
| `attention` | Red/canceled check, failed AI reconciliation, or status details such as `head_mismatch: true` | Inspect the owned PR, fix the root cause in-scope, push, update watcher config head SHA. If `head_mismatch` is true, follow the stop/fetch/inspect branch before any force-push or merge |
| `review_changed` | New review/comment activity since last poll | Inspect comments before any merge decision |
| `ready_for_human_merge` | Checks are green and AI reconciliation passes on the scheduled timer wake | Run `scripts/report_pr_watcher_state.py`, then report readiness or perform the active-builder guarded merge only when explicitly authorized |

## Prompt For Other Builder Sessions

Paste this into any session that should adopt the long-running handoff rules.

```text
You are working in canfieldjuan/ATLAS as a long-running builder session.

Before doing anything, read:
1. AGENTS.md
2. CLAUDE.md
3. docs/SESSION_BOOTSTRAP.md
4. docs/ci_cd_autonomous_coding_map.md
5. docs/long_running_session_watcher_handoff.md
6. The session state file named by ATLAS_SESSION_STATE_FILE if it exists; otherwise create it from docs/SESSION_STATE_TEMPLATE.md

New rules to follow:
- This is a long-running session only for the lane/operator assignment named in this session's state file.
- A PR is yours only if this session's state file lists it under Owned Active PR or PRs This Session May Touch.
- Do not inspect, push to, close, merge, or modify any other open PR unless the operator explicitly reassigns it and you update this session's state file first.
- Record your builder surface in this session's state file: Claude Code native, Codex/local CLI, or other.
- For Claude Code native sessions, subscribe to the owned PR and use Claude Code's native review reactivity plus 30-minute polling. Do not install the local systemd watcher unless the operator explicitly asks for local watcher JSON/log state too.
- For Codex/local CLI sessions, install or refresh a per-session watcher config at ~/.config/atlas-pr-watchers/<session-id>.env only as state production. True autonomous resume requires a separate external wake bridge that starts/resumes Codex with the watcher state.
- Fill the session state hook fields: `Push/review-event hook`, `Timer/poll hook`, `Wake bridge`, `Next timer wake`, `Last watcher state`, and `Standing merge authorization`.
- Record the push/review-event hook in this session's state file only when it wakes the builder session. If no concrete external bridge wakes a Codex/local builder, write `Wake bridge: unavailable`; the scheduled watcher is state-only and the session does not have autonomous review-event wake-up coverage.
- Do not use the scheduled atlas-pr-watch command as the push/review-event bridge unless it has a source-aware event mode that cannot produce merge permission.
- Use `python scripts/codex_wake_bridge.py "${SESSION_ID}" --source event`
  for push/review-event bridges and
  `python scripts/codex_wake_bridge.py "${SESSION_ID}" --source scheduled`
  after scheduled watcher polls that should wake Codex. Event wakes are always
  attention-only; scheduled-ready wakes still require live AGENTS guards.
- A local watcher must poll every 30 minutes and must use AUTO_MERGE="0".
- Codex/local sessions must run `scripts/report_pr_watcher_state.py` on resume before starting the next slice in a long-running arc.
- No auto-merge in the watcher. When the watcher reports ready_for_human_merge, the active builder reports readiness and waits for the operator unless this specific arc has explicit active-builder merge authorization.
- With standing merge authorization recorded in this session's state file, the active builder merges only after a scheduled Claude poll or Codex/local wake bridge reports ready_for_human_merge and the current AGENTS pre-merge guards pass, including review-thread status and merge-conflict/mergeability state.
- Do not merge from a push/review-event wake. If that wake observes green checks, record readiness and wait for the scheduled green-confirmation wake.
- Do not actively poll GitHub for green CI between watcher wake-ups.
- Review/comment events are fast attention only when the operator, Claude Code
  native subscription, or an explicit integration wakes this session. Until a
  Codex/local wake bridge is installed, rely on the scheduled watcher only to
  record `review_changed`.
- If checks are red or review comments are actionable, fix only the owned PR, fix the upstream/root cause within the slice, push with scripts/push_pr.sh, resolve fixed review threads, update the PR body/reconciliation record when needed, and refresh the watcher head SHA.
- If the watcher reports `attention` with `head_mismatch: true`, stop, fetch the remote head, inspect the delta, and do not force-push over another actor's commit.
- If checks are pending, update this session's state file with the current pending list and next poll time.
- Do not start the next slice while the owned PR has unresolved CI, review, AI reconciliation, or merge state.

When you resume:
1. Run gh pr list --state open.
2. Run git log --oneline -15 origin/main.
3. Verify the target PR number, branch, and head SHA match this session's state file.
4. Run scripts/check_session_pr_ownership.py before any PR mutation when PR metadata is known.
5. In Codex/local watcher mode, check the watcher status JSON and journal before deciding whether to fix, wait, or report readiness.

For Codex/local watcher state, use this setup shape:

SESSION_ID="<lane-slug>-<pr-number>"
mkdir -p ~/.config/atlas-pr-watchers ~/.local/state/atlas-pr-watchers
cat > ~/.config/atlas-pr-watchers/${SESSION_ID}.env <<'EOF'
LABEL="<human label>"
REPO_DIR="<absolute repo or worktree path>"
PR="<pr number>"
REPO="canfieldjuan/ATLAS"
SESSION_STATE="<absolute path to SESSION_STATE.<session-id>.local.md>"
HEAD_SHA="<current PR head SHA>"
POLL_MINUTES="30"
AUTO_MERGE="0"
NOTIFY="1"
# Optional, quoted. Leave unset for write-only handoff.
# CODEX_WAKE_COMMAND="codex exec -C <absolute repo or worktree path> -"
EOF

cat > ~/.local/bin/atlas-pr-watch-and-wake <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
session_id="${1:?watcher session id required}"
config="${HOME}/.config/atlas-pr-watchers/${session_id}.env"

if [ ! -f "$config" ]; then
  echo "watcher config not found: $config" >&2
  exit 2
fi

# shellcheck disable=SC1090
source "$config"

if [ -z "${REPO_DIR:-}" ] || [ ! -d "$REPO_DIR" ]; then
  echo "invalid REPO_DIR for ${session_id}: ${REPO_DIR:-}" >&2
  exit 2
fi

~/.local/bin/atlas-pr-watch "${session_id}"
cd "$REPO_DIR"
python scripts/codex_wake_bridge.py "${session_id}" --source scheduled
EOF
chmod +x ~/.local/bin/atlas-pr-watch-and-wake

mkdir -p ~/.config/systemd/user/atlas-pr-watch@.service.d
cat > ~/.config/systemd/user/atlas-pr-watch@.service.d/wake-bridge.conf <<'EOF'
[Service]
ExecStart=
ExecStart=%h/.local/bin/atlas-pr-watch-and-wake %i
EOF

~/.local/bin/atlas-pr-watch-and-wake "${SESSION_ID}"
systemctl --user daemon-reload
systemctl --user enable --now "atlas-pr-watch@${SESSION_ID}.timer"
```

## Current Codex/Local Watcher Example

The first live local watcher instance used:

```text
SESSION_ID=ci-cd-autonomous-map-1963
LABEL="CI/CD autonomous map #1963"
PR=1963
AUTO_MERGE=0
```

That watcher recorded an open Codex reconciliation thread, the active builder
fixed the doc issue, pushed a new head, resolved the outdated thread, and the
watcher later reported `ready_for_human_merge` without merging.

The #1968 run added the standing-authorization case for an already-active
builder: the operator explicitly authorized the active builder to merge after a
scheduled green confirmation. The watcher still did not merge; it only produced
the `ready_for_human_merge` snapshot. The builder then ran the ready-state
reporter, re-ran the ownership/head/check/thread guards, merged, disabled the
timer, and tore down the worktree.

The #1973 incident proved why this is enforced: a stale local watcher path still
carried merge behavior even though docs said no auto-merge. From this point on,
watcher merge authority is a blocking audit failure, and desktop notifications
are not treated as a builder wake-up.

## Pattern Reports

For a long-running arc that spans multiple PRs, use
`docs/long_running_agent_monitoring_spec.md` at merge checkpoints or arc close
to record what the watcher and review loop learned: PR cycle time,
red-to-green loops, pushes per PR, recurring CI failures, review finding
classes, stale branch count, unresolved thread count, and codification
decisions.

The pattern report is evidence, not authority. It can propose AGENTS updates,
audits, tests, or watcher changes for repeated failure classes, but it does not
merge PRs or change the scheduled-ready-only merge rule.

## New Repo Playbook

For future repos, use `docs/autonomous_coding_repo_playbook.md` instead of
copying Atlas wholesale. It lists the minimum portable contracts, the
Atlas-specific pieces to avoid copying blindly, and the smaller-repo CI stack
that preserves the same plan/review/reconciliation discipline.

# Long-Running Session Watcher Handoff

Issue: #1962

Use this handoff when the operator wants a builder session to keep an owned PR
moving while they are away. It complements `AGENTS.md` section 3c.1 and the CI
map in `docs/ci_cd_autonomous_coding_map.md`.

## What Changed

Long-running sessions now have two durable responsibilities:

1. Keep `SESSION_STATE.local.md` current for the owned lane and PR.
2. Use two signal paths after each PR open or push: a required
   push/review-event hook when the environment provides one, and the autonomous
   30-minute scheduled green-confirmation hook.

The watcher is intentionally local and per session. One session equals one
watcher config. A second session should create a second config and timer rather
than reuse another session's watcher.

## Two Signal Paths

Long-running sessions should not actively poll GitHub in a loop. They wait for
signals, then take the narrow action that signal allows. Today, this repo
provides the scheduled watcher; a push/review-event hook must be supplied by the
operator environment or recorded as unavailable in `SESSION_STATE.local.md`.

| Signal | Source | Builder action | Autonomous today? | Merge allowed? |
|---|---|---|---|---|
| Push/review-event attention | External bridge for GitHub push, review thread, review event, or reconciliation event; operator "review is up" signal if no bridge exists | Inspect only the owned PR, read thread-aware review state, fix actionable feedback in scope, and record readiness if already green | Only when the environment supplies the bridge | No |
| Scheduled green confirmation | The local `atlas-pr-watch@<session>.timer` 30-minute wake-up | If the scheduled state is `ready_for_human_merge`, run the AGENTS ownership/head/check/thread guards | Yes | Only with explicit standing authorization for this arc |

The push/review-event path exists to reduce red-review latency when something
wakes the builder. It is an attention signal, not a merge signal. If no concrete
bridge is installed, record `push/review-event hook: unavailable` in
`SESSION_STATE.local.md`; the scheduled watcher remains responsible for catching
`review_changed` while the operator is away.

The scheduled hook is the canonical merge gate. A push/review-event wake that
observes a clean PR records readiness, leaves the timer armed, and waits for the
scheduled `ready_for_human_merge` confirmation before any standing-authorized
merge.

## Local Watcher Files

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

If `~/.local/bin/atlas-pr-watch` is missing, stop and ask the operator. Do not
recreate a merge-capable watcher from scratch. The watcher must stay read-only
with respect to GitHub merges.

## Push/Review-Event Hook

AGENTS.md requires a push/review-event wake hook for fully immediate
long-running operation. That hook is outside this repo unless the operator has
provided an integration. Use this concrete contract:

1. Record the hook in `SESSION_STATE.local.md` as
   `Push/review-event hook: <name and trigger>`.
2. Configure the external bridge to wake the builder session on new pushes,
   review threads, review events, and reconciliation events for the owned PR. Do
   not use the scheduled
   `~/.local/bin/atlas-pr-watch "${SESSION_ID}"` command as the event bridge
   unless it has a source-aware event mode that cannot grant merge permission.
3. If no such bridge exists, record `Push/review-event hook: unavailable` and
   rely on the scheduled watcher for autonomous review-change detection.
   Operator-only notifications are a manual fallback, not a recorded
   push/review-event hook.

The unavailable state is safe but not immediate. Do not describe that session as
having quick review-event wake-up coverage.

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
  `SESSION_STATE.local.md` before merging.
- Pending states write a handoff but do not run the optional command by default.
  Do not replace the watcher timer with an in-chat polling loop.

This gives the two-hook shape without making the watcher merge-capable: event
hooks can call the bridge with `--source event`; the 30-minute green-confirmation
timer can call it with `--source scheduled`.

## Safety Rules

- No auto-merge. The watcher refuses `AUTO_MERGE=1`.
- A green PR becomes `ready_for_human_merge`; it does not merge itself.
- If the operator gives explicit standing authorization for an arc, the builder
  may merge after a scheduled `ready_for_human_merge` wake-up, but only after
  re-running the AGENTS pre-merge guards: open PR list, `origin/main` log,
  ownership guard, matching head SHA, current checks, review-thread status,
  live reconciliation, and merge-conflict/mergeability state.
- Standing authorization applies only to the scheduled green-confirmation wake,
  never to a push/review-event attention wake.
- Review/comment events do not currently wake the local builder session by
  themselves. Treat them as fast attention only when the operator or another
  explicit integration wakes the session; otherwise record the event hook as
  unavailable and use the scheduled watcher as the autonomous fallback.
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

## Setup For A Session

Pick a stable session id:

```bash
SESSION_ID="<lane-slug>-<pr-number>"
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
SESSION_STATE="<absolute path to SESSION_STATE.local.md>"
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
| `ready_for_human_merge` | Checks are green and AI reconciliation passes on the scheduled timer wake | If this arc has standing merge authorization and this state came from the scheduled timer, run the AGENTS merge guards and merge; otherwise report readiness and wait |

## Prompt For Other Builder Sessions

Paste this into any session that should adopt the long-running watcher.

```text
You are working in canfieldjuan/ATLAS as a long-running builder session.

Before doing anything, read:
1. AGENTS.md
2. CLAUDE.md
3. docs/SESSION_BOOTSTRAP.md
4. docs/ci_cd_autonomous_coding_map.md
5. docs/long_running_session_watcher_handoff.md
6. SESSION_STATE.local.md if it exists; otherwise create it from docs/SESSION_STATE_TEMPLATE.md

New rules to follow:
- This is a long-running session only for the lane/operator assignment named in SESSION_STATE.local.md.
- A PR is yours only if SESSION_STATE.local.md lists it under Owned Active PR or PRs This Session May Touch.
- Do not inspect, push to, close, merge, or modify any other open PR unless the operator explicitly reassigns it and you update SESSION_STATE.local.md first.
- After every PR open or push, install or refresh a per-session watcher config at ~/.config/atlas-pr-watchers/<session-id>.env.
- Fill the SESSION_STATE.local.md hook fields: `Push/review-event hook`, `Timer hook`, `Next timer wake`, `Last watcher state`, and `Standing merge authorization`.
- Record the push/review-event hook in SESSION_STATE.local.md only when it wakes the builder session. If no concrete external bridge wakes the builder, write `Push/review-event hook: unavailable`; the scheduled watcher is then the autonomous fallback and the session does not have immediate review-event wake-up coverage.
- Do not use the scheduled atlas-pr-watch command as the push/review-event bridge unless it has a source-aware event mode that cannot produce merge permission.
- Use `python scripts/codex_wake_bridge.py "${SESSION_ID}" --source event`
  for push/review-event bridges and
  `python scripts/codex_wake_bridge.py "${SESSION_ID}" --source scheduled`
  after scheduled watcher polls that should wake Codex. Event wakes are always
  attention-only; scheduled-ready wakes still require live AGENTS guards.
- The watcher must poll every 30 minutes and must use AUTO_MERGE="0".
- No auto-merge in the watcher. When the watcher reports ready_for_human_merge, report readiness and wait for the operator unless this specific arc has explicit standing merge authorization.
- With standing merge authorization recorded in SESSION_STATE.local.md, merge only after a scheduled watcher wake-up reports ready_for_human_merge and the current AGENTS pre-merge guards pass, including review-thread status and merge-conflict/mergeability state.
- Do not merge from a push/review-event wake. If that wake observes green checks, record readiness and wait for the scheduled green-confirmation wake.
- Do not actively poll GitHub for green CI between watcher wake-ups.
- Review/comment events are fast attention only when the operator or an
  explicit integration wakes this session. Until a local review-event bridge is
  installed, rely on the scheduled watcher to catch `review_changed`.
- If checks are red or review comments are actionable, fix only the owned PR, fix the upstream/root cause within the slice, push with scripts/push_pr.sh, resolve fixed review threads, update the PR body/reconciliation record when needed, and refresh the watcher head SHA.
- If the watcher reports `attention` with `head_mismatch: true`, stop, fetch the remote head, inspect the delta, and do not force-push over another actor's commit.
- If checks are pending, update SESSION_STATE.local.md with the current pending list and next poll time.
- Do not start the next slice while the owned PR has unresolved CI, review, AI reconciliation, or merge state.

When you resume:
1. Run gh pr list --state open.
2. Run git log --oneline -15 origin/main.
3. Verify the target PR number, branch, and head SHA match SESSION_STATE.local.md.
4. Run scripts/check_session_pr_ownership.py before any PR mutation when PR metadata is known.
5. Check the watcher status JSON and journal before deciding whether to fix, wait, or report readiness.

Use this watcher setup shape:

SESSION_ID="<lane-slug>-<pr-number>"
mkdir -p ~/.config/atlas-pr-watchers ~/.local/state/atlas-pr-watchers
cat > ~/.config/atlas-pr-watchers/${SESSION_ID}.env <<'EOF'
LABEL="<human label>"
REPO_DIR="<absolute repo or worktree path>"
PR="<pr number>"
REPO="canfieldjuan/ATLAS"
SESSION_STATE="<absolute path to SESSION_STATE.local.md>"
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

## Current Example

The first live instance used:

```text
SESSION_ID=ci-cd-autonomous-map-1963
LABEL="CI/CD autonomous map #1963"
PR=1963
AUTO_MERGE=0
```

That watcher caught an open Codex reconciliation thread, the builder fixed the
doc issue, pushed a new head, resolved the outdated thread, and the watcher
reported `ready_for_human_merge` without merging.

The #1968 run added the standing-authorization case: the operator explicitly
authorized the builder to merge after a scheduled green confirmation. The
watcher still did not merge; it only produced the `ready_for_human_merge`
snapshot. The builder then re-ran the ownership/head/check/thread guards,
merged, disabled the timer, and tore down the worktree.

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

# Long-Running Session Watcher Handoff

Issue: #1962

Use this handoff when the operator wants a builder session to keep an owned PR
moving while they are away. It complements `AGENTS.md` section 3c.1 and the CI
map in `docs/ci_cd_autonomous_coding_map.md`.

## What Changed

Long-running sessions now have two durable responsibilities:

1. Keep `SESSION_STATE.local.md` current for the owned lane and PR.
2. Run a lane-local PR watcher every 30 minutes after each PR open or push.

The watcher is intentionally local and per session. One session equals one
watcher config. A second session should create a second config and timer rather
than reuse another session's watcher.

## Local Watcher Files

These files are local machine infrastructure, not committed repo files:

| Path | Purpose |
|---|---|
| `~/.local/bin/atlas-pr-watch` | One-shot watcher executable |
| `~/.config/systemd/user/atlas-pr-watch@.service` | User systemd service template |
| `~/.config/systemd/user/atlas-pr-watch@.timer` | User systemd timer template, every 30 minutes |
| `~/.config/atlas-pr-watchers/<session-id>.env` | One config per builder session |
| `~/.local/state/atlas-pr-watchers/<session-id>.json` | Latest machine-readable watcher status |
| `~/.local/state/atlas-pr-watchers/<session-id>.log` | Append-only watcher log |

If `~/.local/bin/atlas-pr-watch` is missing, stop and ask the operator. Do not
recreate a merge-capable watcher from scratch. The watcher must stay read-only
with respect to GitHub merges.

## Safety Rules

- No auto-merge. The watcher refuses `AUTO_MERGE=1`.
- A green PR becomes `ready_for_human_merge`; it does not merge itself.
- The watcher may alert on red CI, pending CI, new review activity, head SHA
  mismatch, or failed AI reconciliation.
- A builder may fix only the owned PR and only the files allowed by the active
  slice or PR-fix-mode block.
- After merge, disable that PR's timer and tear down only that session's owned
  worktree/branch.

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
EOF
```

Run one manual poll first:

```bash
~/.local/bin/atlas-pr-watch "${SESSION_ID}"
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
| `attention` | Red/canceled check, head mismatch, or failed AI reconciliation | Inspect the owned PR, fix the root cause in-scope, push, update watcher config head SHA |
| `review_changed` | New review/comment activity since last poll | Inspect comments before any merge decision |
| `ready_for_human_merge` | Checks are green and AI reconciliation passes | Report readiness; do not merge unless the operator explicitly authorizes it |

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
- The watcher must poll every 30 minutes and must use AUTO_MERGE="0".
- No auto-merge. When the watcher reports ready_for_human_merge, report readiness and wait for the operator unless this specific arc has explicit merge authorization.
- If checks are red or review comments are actionable, fix only the owned PR, fix the upstream/root cause within the slice, push with scripts/push_pr.sh, resolve fixed review threads, update the PR body/reconciliation record when needed, and refresh the watcher head SHA.
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
EOF
~/.local/bin/atlas-pr-watch "${SESSION_ID}"
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

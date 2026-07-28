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

The watcher executable is an installed copy of repo-owned
`scripts/pr_watcher.py`; its configs and output remain intentionally local and
per session. One Codex/local session equals one watcher config. A second
Codex/local session should create a second config and timer rather than reuse
another session's watcher. Claude Code native sessions do not need this local
watcher path.

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
| `~/.local/bin/atlas-pr-watch` | Installed repo-owned one-shot producer (`scripts/pr_watcher.py`) |
| `~/.config/systemd/user/atlas-pr-watch@.service` | User systemd service template |
| `~/.config/systemd/user/atlas-pr-watch@.timer` | User systemd timer template, every 30 minutes |
| `~/.config/atlas-pr-watchers/<session-id>.env` | One config per builder session |
| `~/.local/state/atlas-pr-watchers/<session-id>.json` | Latest machine-readable watcher status |
| `~/.local/state/atlas-pr-watchers/<session-id>.wake.json` | Latest Codex wake-bridge handoff metadata |
| `~/.local/state/atlas-pr-watchers/<session-id>.wake.md` | Pasteable/resumable Codex wake prompt |
| `~/.local/state/atlas-pr-watchers/<session-id>.log` | Append-only watcher log |

If a Codex/local session is using local watcher state and
`~/.local/bin/atlas-pr-watch` is missing or drifted, run the repo-owned
installer and its `--check` mode. Do not recreate a watcher from ad hoc local
source. The watcher must stay read-only with respect to GitHub merges.

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

The watcher records state; it does not wake Codex by itself. Use the installed
bridge copy to convert an existing watcher snapshot into a resumable handoff.
For one-off development checks, the repo script is equivalent:

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
interpolated into a shell command. Do not use no-approval/full-filesystem Codex
flags in this config unless the watched PR, watcher config, and PR metadata are
all trusted; watcher-sourced text is treated as untrusted prompt input.

Wake-source rules:

- `--source event` is attention-only. It can wake Codex to inspect review
  activity or a new push, but it must not merge, even if the watcher snapshot is
  green. Pending, ready, review-changed, and failure snapshots are actionable
  as inspection/fix handoffs only; scheduled green-confirmation is still the
  only wake source that may proceed to merge consideration after live guards.
- `--source scheduled` may classify `ready_for_human_merge` as
  `scheduled-ready` only when the snapshot carries a version-1 readiness proof
  for the same open, non-draft head: at least one required check, all required
  checks complete/green, complete review-thread pagination, zero unresolved
  non-outdated Codex connector threads, complete Codex review pagination with
  at least one current-head Codex connector review, no changes-requested
  decision, and a clean merge state. Missing or contradictory proof becomes
  `attention` and lists its blockers. `scheduled-ready` is still only
  permission to run the AGENTS merge guards; the resumed builder also needs
  explicit standing merge authorization in this session's state file before
  merging.
- Pending states write a handoff but do not run the optional command by default.
  Do not replace the watcher timer with an in-chat polling loop.

The version-1 snapshot proof is:

```json
{
  "readiness": {
    "version": 1,
    "evaluated_head_sha": "<same as pr.headRefOid>",
    "required_check_count": 3,
    "required_checks_complete": true,
    "required_check_failures": [],
    "required_check_pending": [],
    "review_threads_complete": true,
    "review_thread_pages_fetched": 1,
    "unresolved_review_threads": [],
    "codex_reviews_complete": true,
    "codex_review_pages_fetched": 1,
    "codex_head_review_count": 1,
    "review_decision": "<same as pr.reviewDecision>",
    "merge_state_status": "CLEAN"
  }
}
```

This gives the two-hook shape without making the watcher merge-capable: event
hooks can call the bridge with `--source event`; the 30-minute green-confirmation
timer can call it with `--source scheduled`.

## Safety Rules

- No auto-merge. Truthy watcher auto-merge config is unsafe and must surface as
  attention, not an action path.
- A local producer may label a PR `ready_for_human_merge`, but the bridge and
  reporter fail closed unless the versioned readiness proof above is complete.
  Legacy snapshots therefore remain attention-only until their producer is
  upgraded. No watcher snapshot merges by itself.
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
- At a technical fork, take the durable engineering fix that will not break
  later, document the reasoning in the PR, and keep going. Do not present the
  shortcut as an equal option. Defer only decisions that are genuinely
  operator-owned, such as product positioning, customer-facing policy, spend,
  credentials, production data, irreversible action, scope ownership, or risk
  tolerance. Deferred operator decisions go to a GitHub issue and notification
  path in the follow-up slice; they are not a blocking chat-stop when other safe
  queued work remains.
- After a guarded merge, use `scripts/codex_issue_queue.py next --lane <lane>`
  to find the next issue-backed slice for the same lane. The queue source is
  GitHub Issues with the `codex` label plus an `Autonomy lane: <lane>` marker
  and optional `Autonomy priority: <int>` marker. Issue-body markers are trusted
  only on `codex`-labeled issues; comment markers are trusted only from GitHub
  author associations with repository write-level trust. Do not infer the next
  slice from chat memory.
- If a fork is genuinely operator-owned, record it with
  `scripts/codex_issue_queue.py defer --issue <n> --lane <lane> --reason
  "<why this belongs to the operator>"`. This writes a local email-ready
  artifact under
  `~/.local/state/atlas-pr-watchers/operator-defers/`; it does not send email.
  The GitHub issue is then labeled `deferred` and receives a quoted defer
  comment so multiline operator text cannot become queue-control markers.

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

Install the repo-owned watcher producer, bridge wrapper, trusted bridge copy,
AI-reconciliation checker and parser dependency, and systemd drop-in through
the installer. The watcher and wrapper invoke those installed copies rather
than executing scripts from the watched PR worktree. One systemd template can
safely serve multiple sessions without baking one worktree path into every
timer.

For any already-enabled local watcher, rerun the installer after pulling a
watcher or reconciliation change and before trusting a new
`ready_for_human_merge` snapshot. Merging repository source does not upgrade the
copies under `~/.local/bin`; the active systemd wrapper continues to execute the
previous installed producer until this reinstall/check step succeeds.

```bash
python scripts/install_codex_wake_bridge.py --reload-systemd
python scripts/install_codex_wake_bridge.py --check
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
| `pending` | At least one check is still pending and no new review/comment activity was observed | Record the next poll; do not ask the operator to babysit CI |
| `attention` | Red/canceled check, failed AI reconciliation, or status details such as `head_mismatch: true` | Inspect the owned PR, fix the root cause in-scope, push, update watcher config head SHA. If `head_mismatch` is true, follow the stop/fetch/inspect branch before any force-push or merge |
| `review_changed` | New review/comment activity since last poll, including while checks are pending | Inspect comments before any merge decision |
| `ready_for_human_merge` | The snapshot label and version-1 proof agree: same open/non-draft head, required checks complete/green, all thread pages fetched, zero unresolved non-outdated Codex connector threads, complete Codex review pagination, at least one current-head Codex connector review, no changes requested, clean merge state | Run `scripts/report_pr_watcher_state.py`; missing/contradictory proof is reported as attention. Otherwise report readiness or perform the active-builder guarded merge only when explicitly authorized and after fresh live guards |

The installed producer reads branch protection's required-context inventory,
compares it with `gh pr checks --required`, fetches every GraphQL
`reviewThreads` page, fetches every current-head Codex connector review page,
and reads PR metadata again after those calls. This prevents a required context
that has not reported yet from disappearing from the observed set and prevents
review pagination from proving readiness for a head that is no longer current.
A changed head, empty/malformed required policy, unreported required context,
incomplete pagination, unresolved non-outdated Codex connector thread, missing
current-head Codex connector review, or GitHub read error cannot produce a
ready proof. The JSON snapshot is replaced atomically so the bridge/reporter
cannot consume a partial file.
Live AI reconciliation runs from the exact checker and parser sources installed
beside the watcher; it never executes the watched PR worktree's checker.

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
  after scheduled watcher polls that should wake Codex. Installed systemd
  wrappers should call the installed bridge copy, not the watched PR worktree's
  script. Event wakes are always attention-only; scheduled-ready wakes still
  require live AGENTS guards.
- A local watcher must poll every 30 minutes and must use AUTO_MERGE="0".
- After any watcher, bridge, or reconciliation checker source change, rerun
  `python scripts/install_codex_wake_bridge.py --reload-systemd` and
  `python scripts/install_codex_wake_bridge.py --check` before relying on the
  installed watcher state. Existing timers execute installed copies from
  `~/.local/bin`, not the repository files.
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

python scripts/install_codex_wake_bridge.py --reload-systemd
python scripts/install_codex_wake_bridge.py --check
~/.local/bin/atlas-pr-watch-and-wake "${SESSION_ID}"
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

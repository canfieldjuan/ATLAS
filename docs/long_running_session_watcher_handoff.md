# Long-Running Session Watcher Handoff

Issue: #1962

Use this handoff when the operator wants a builder session to keep an owned PR
moving while they are away. It complements `AGENTS.md` section 3c.1 and the CI
map in `docs/ci_cd_autonomous_coding_map.md`.

Important mode split:

- **Claude Code native sessions** should use Claude Code's PR subscription and
  review reactivity. A later platform/external activation takes one exact-head
  snapshot; the active model does not remain in a polling loop. Do not force
  those sessions onto the local systemd `atlas-pr-watch` timer unless the
  operator explicitly asks for local state files too.
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
   subscription, a Codex wake bridge, or local watcher state-only.

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
| Claude Code native | Claude Code PR subscription/review reactivity plus a later platform/external activation | Claude Code resumes as the active builder, takes one exact-head snapshot, acts on new evidence, and yields again when pending/unchanged | Yes, when the subscription can reactivate the builder | Active builder only after explicit source-matched operator authorization and fresh guards |
| Codex wake bridge | External wrapper that starts/resumes Codex with watcher state | Fresh/active Codex reads this session's state file, runs `scripts/report_pr_watcher_state.py`, then fixes, waits, reports ready, or runs guarded merge if authorized | Only when the bridge exists | Active builder only after explicit operator authorization and fresh guards |
| Local watcher state-only | `atlas-pr-watch@<session>.timer` writes JSON/log state every 30 minutes | No agent wakes automatically; the next active agent consumes the state with `scripts/report_pr_watcher_state.py` | No | No |
| Operator signal | Human says "review is up", "green", or "merge" | Active builder inspects the owned PR and runs the same guards | Manual | Active builder only after explicit operator authorization and fresh guards |

The push/review-event path exists to reduce red-review latency when something
wakes the builder. It is an attention signal, not a merge signal. If no concrete
bridge exists for Codex/local sessions, record `Wake bridge: unavailable` in
this session's state file; the scheduled watcher remains only a state recorder
until an active agent consumes its output.

No model-driven poll is the canonical readiness signal. A subscription,
operator signal, webhook, or external non-model wake activates the builder, and
that activation takes one exact-head snapshot before acting or yielding. A
guarded merge additionally requires the activation source to satisfy the
authorization recorded in session state.

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
session's watcher config. Use the installed runner, not a bare `codex exec`:

```bash
CODEX_WAKE_COMMAND="'/home/<you>/.local/bin/atlas-codex-wake-run' --watcher-id '<session-id>' --repo-dir '/home/<you>/path/to/repo'"
```

A wake can also be given its own Codex profile, which is where nearly all of a
wake's cost lives. Measured on this host with one identical prompt, an
interactive profile cost 75,685 tokens on the first wake and added 87,509 per
wake after, while the whole two-turn conversation was about 90 tokens. The rest
is context re-injected every turn: a list of plugins that are not installed, a
capped copy of `AGENTS.md`, memory-folder guidance, and a skills catalogue sent
twice. Under an isolated profile the same pair cost 27,495 and 13,862.

```bash
CODEX_WAKE_COMMAND="'/home/<you>/.local/bin/atlas-codex-wake-run' --watcher-id '<session-id>' --repo-dir '<repo>' --codex-home '/home/<you>/.codex-wake' --agent-home '/home/<you>/.codex-wake-home'"
```

`--codex-home` sets `CODEX_HOME`, which supplies config, memories and built-in
skills. `--agent-home` sets `HOME`, because the shared skills catalogue lives at
`$HOME/.agents/skills` and is not under `CODEX_HOME`. They are independent: each
takes effect alone, and isolating only one is recorded in the wake log as
partial, because `CODEX_HOME` alone still carried all 26 shared skills.

A wake `HOME` still needs the tools the agent uses. Symlink `.gitconfig`,
`.ssh` and `.config/gh` into it from the real home, or the woken agent loses its
git identity and `gh` auth. The runner does not probe for them.

Both arguments must be non-empty absolute paths; anything else is rejected
before a turn starts. An empty value is not a profile, because Codex treats an
empty `CODEX_HOME` as unset and falls back to `$HOME/.codex`, and a relative or
`~` path would be resolved against a directory the watcher config does not
name. Beyond that, the runner does not validate a profile. Codex already refuses a nonexistent
`CODEX_HOME` before any model call, and fails an unauthenticated or
non-writable one without billing tokens, so a check here would duplicate that
and could reject a profile Codex accepts. Every launching turn logs which
profile it used, including when none is configured.

A stored thread id belongs to the `CODEX_HOME` that created it, so each
effective Codex home keeps its own thread file, named
`<session-id>.codex-thread.<digest>` from a digest of that home. That includes
a wake given no profile arguments: its home is whatever `CODEX_HOME` or
`$HOME/.codex` it inherits, so if a watcher's environment changes, the new home
starts a fresh thread instead of resuming one that home does not hold. The home
is the directory Codex actually uses, resolved at wake time the way `codex
doctor` reports it: a relative value is taken against the wake's `--repo-dir`,
and symlinks are followed. Two spellings or aliases of one directory therefore
share one arc, and a profile link retargeted to another directory starts that
directory's own arc; pointing it back resumes the first. Do not retarget a
profile link while one of its wakes is running. A path that is not valid UTF-8
works, and is shown with backslash escapes in the log. The wake log's
`profile ...` line names the resolved home and the file in use.

A watcher created before this change has a single `<session-id>.codex-thread`
file. The runner never reads it, so that watcher's next wake starts one fresh
thread under its home. Run the reset below once per existing watcher after
upgrading to remove the old file.

To reset a watcher, at post-merge teardown or when resumes keep failing, run
the runner's reset command rather than deleting files by pattern. Pass the same
`--state-dir` as the watcher's `CODEX_WAKE_COMMAND`, or omit it only if that
command omits it too; a reset of a different directory finds nothing and
prints `no thread files for <session-id> in <dir>`:

```bash
~/.local/bin/atlas-codex-wake-run --watcher-id '<session-id>' --state-dir '<same state dir as the wake command>' --reset-threads
```

It removes exactly that watcher's thread files, including quarantined ones and
the pre-profile file, and it takes the watcher's wake lock first, so it cannot
interleave with a wake in flight. A filename glob is not safe here: watcher ids
may contain dots and hyphens, so `<session-id>.codex-thread*` also matches a
different watcher named `<session-id>.codex-thread-<anything>`.

Use absolute paths, and quote each one individually as shown. The bridge does
not run this through a shell: it `shlex.split`s the value and hands the argv
straight to `subprocess.run`. Two consequences follow. A `${HOME}` or `~` stays
literal, so the wake dies with `FileNotFoundError` instead of starting a Codex
turn. And the outer quotes around the whole value do not survive to protect the
individual arguments, so a repo or worktree path containing a space splits into
separate argv entries and the runner rejects the stray ones. The inner quotes
are what keep such a path intact.

The command receives the generated prompt on stdin. The prompt text is not
interpolated into a shell command. Do not use no-approval/full-filesystem Codex
flags in this config unless the watched PR, watcher config, and PR metadata are
all trusted; watcher-sourced text is treated as untrusted prompt input.

### Why the runner, not a bare `codex exec`

`atlas-codex-wake-run` is the installed copy of `scripts/codex_wake_run.py`. It
exists because a bare `codex exec` per wake starts a **fresh thread**: the
woken agent has no memory of the arc it is continuing, and the operator pays
full cold-start context on every wake. The runner keeps one Codex thread per
watcher id, recorded at
`~/.local/state/atlas-pr-watchers/<session-id>.codex-thread`, and resumes it.

It also pins the argv to what the installed Codex actually accepts. The
hand-written local script this replaces passed `--ask-for-approval`, which
current Codex rejects outright, so every wake it ever attempted failed with
`error: unexpected argument '--ask-for-approval' found` and nothing surfaced
that. Because the runner is installed and drift-checked by
`scripts/install_codex_wake_bridge.py --check`, and its argv shape is pinned by
`tests/test_codex_wake_run.py`, that class of silent breakage is now caught.

Two details the runner encodes, verified against codex-cli 0.155.1: `codex exec`
accepts `-C/--cd` and `-s/--sandbox` but `codex exec resume` accepts neither, so
the working directory is passed as the subprocess cwd and the sandbox as
`-c sandbox_mode="<value>"`, which both subcommands take. The sandbox defaults
to `workspace-write` rather than full access, because the wake prompt is built
from PR and review text that this document already classifies as untrusted.

Verify wiring without spending tokens:

```bash
~/.local/bin/atlas-codex-wake-run --watcher-id <session-id> --repo-dir <repo-dir> --dry-run
```

Each wake appends its mode, argv, the turn's token usage, and the agent's final
message to `~/.local/state/atlas-pr-watchers/<session-id>.codex-wake.log`, so
wake cost and outcome are readable rather than inferred. A turn that Codex
never priced is not reported as a success: the log says `usage=unavailable`
and the wake exits 81, because a wake whose cost is unknown is the failure
this runner was built to make visible. The work itself still happened, so the
thread id and the agent message are kept and the arc continues. Concurrent wakes are
serialized by a lock: a wake that arrives while one is in flight waits for the
lock and then runs its own prompt. It is not dropped, because the running turn
may already have taken its snapshot of the PR and cannot see a review posted
after that point. A wake that never gets the lock exits non-zero rather than
reporting a success it did not perform.

Each turn runs under a supervisor that leads its own process group, so the
Codex process and the ordinary commands it starts do not outlive the wake lock.
A descendant that calls `setsid` leaves that group and is not covered; issue
#2526 tracks containment a descendant cannot opt out of.

A merged PR leaves its thread state behind: one thread file per Codex home the
watcher ran under. Run the reset command above, with the watcher's own
`--state-dir`, during the post-merge teardown in AGENTS 3c.1 so the next PR
on that watcher id does not resume a finished arc. The reset takes the wake
lock, so it waits for any wake in flight rather than racing it.

Wake-source rules:

- `--source event` is attention-only. It can wake Codex to inspect review
  activity or a new push, but it must not merge, even if the watcher snapshot is
  green. Pending, ready, review-changed, and failure snapshots are inspection or
  fix handoffs only. Only `--source scheduled` can produce the bridge's
  `scheduled-ready` classification, and only source-matched authorization can
  use that classification for merge consideration after live guards.
- `--source scheduled` may classify `ready_for_human_merge` as
  `scheduled-ready` only when the snapshot carries a version-1 readiness proof
  for the same open, non-draft head: at least one required check, all required
  checks complete/green, complete Codex review pagination with at least one
  exact-head Codex attestation and no exact-head change request, complete
  review-thread pagination, zero unresolved non-outdated Codex connector
  threads, and a clean merge state.
  Missing or contradictory proof becomes
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
    "codex_review_pages_fetched": 2,
    "codex_head_review_count": 1,
    "codex_changes_requested": false,
    "review_decision": "<same as pr.reviewDecision>",
    "merge_state_status": "CLEAN"
  }
}
```

This gives the two-hook shape without making the watcher merge-capable: event
hooks can call the bridge with `--source event`; an optional external non-model
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
- Claude Code native sessions use Claude Code's PR subscription; each later
  platform or external activation takes one exact-head snapshot and yields when
  state is pending or unchanged. Local systemd watcher setup is optional, not
  required.
- If the operator gives explicit standing authorization for an arc, the active
  builder may merge after a source-matched `ready_for_human_merge` wake-up, but
  only after re-running the AGENTS pre-merge guards: open PR list,
  `origin/main` log, ownership guard, matching head SHA, current checks,
  review-thread status, live reconciliation, and merge-conflict/mergeability
  state.
- Standing authorization applies only to the wake source it names; a
  `scheduled-ready-only` authorization never applies to a push/review-event wake.
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
# CODEX_WAKE_COMMAND="'/home/<you>/.local/bin/atlas-codex-wake-run' --watcher-id '<session-id>' --repo-dir '<absolute repo or worktree path>'"
# Absolute paths only: the bridge shlex.splits this and never uses a shell.
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

Enable the optional external non-model timer:

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
| `pending` | A required check or complete exact-head Codex review is still pending and no new actionable activity was observed | Record the latest snapshot; do not keep the model active or ask the operator to babysit CI |
| `attention` | Red/canceled check, failed AI reconciliation, or status details such as `head_mismatch: true` | Inspect the owned PR, fix the root cause in-scope, push, update watcher config head SHA. If `head_mismatch` is true, follow the stop/fetch/inspect branch before any force-push or merge |
| `review_changed` | New review/comment activity since the last external snapshot, including while checks are pending | Inspect comments before any merge decision |
| `ready_for_human_merge` | The snapshot label and version-1 proof agree: same open/non-draft head, required checks complete/green, complete exact-head Codex review evidence with at least one attestation, all thread pages fetched, zero unresolved non-outdated Codex connector threads, clean merge state | Run `scripts/report_pr_watcher_state.py`; missing/contradictory proof is reported as attention. Otherwise report readiness or perform the active-builder guarded merge only when explicitly source-authorized and after fresh live guards |

The installed producer reads branch protection's required-context inventory,
then replaces that expected set with `origin/main:ci/gates.yml` parsed through
`origin/main:scripts/check_required_status_checks.py` when the trusted registry
exists. It compares the expected set with `gh pr checks --required`, fetches
every GraphQL `reviewThreads` page, and reads PR metadata again after those
calls. This prevents a required context that has not reported yet from
disappearing from the observed set. A changed head, empty/malformed required
policy, unreported required context, incomplete review/thread pagination,
absent or stale exact-head Codex attestation, unresolved non-outdated Codex
connector thread, or GitHub read error cannot produce a ready proof. The JSON
snapshot is replaced atomically so the bridge/reporter
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
- For Claude Code native sessions, subscribe to the owned PR and use native review reactivity. On each later platform/external activation, take one exact-head snapshot and yield again if it is pending or unchanged. Do not install the local systemd watcher unless the operator explicitly asks for local watcher JSON/log state too.
- For Codex/local CLI sessions, install or refresh a per-session watcher config at ~/.config/atlas-pr-watchers/<session-id>.env only as state production. True autonomous resume requires a separate external wake bridge that starts/resumes Codex with the watcher state.
- Fill the session state hook fields: `Push/review-event hook`, `Timer/poll hook`, `Wake bridge`, `Next external wake`, `Last watcher state`, and `Standing merge authorization`.
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
- With standing merge authorization recorded in this session's state file, the active builder merges only when the current activation source satisfies that recorded authorization, a fresh exact-head snapshot reports ready_for_human_merge, and the current AGENTS pre-merge guards pass, including review-thread status and merge-conflict/mergeability state. Scheduled-ready-only authority requires an external scheduled wake.
- Do not merge from a push/review-event wake. If that wake observes green checks, record readiness and end the model turn; a later authorized activation takes its own fresh snapshot.
- Do not actively poll GitHub for green CI between watcher wake-ups.
- Review/comment events are fast attention only when the operator, Claude Code
  native subscription, or an explicit integration wakes this session. Until a
  Codex/local wake bridge is installed, rely on the scheduled watcher only to
  record `review_changed`.
- If checks are red or review comments are actionable, fix only the owned PR, fix the upstream/root cause within the slice, push with scripts/push_pr.sh, resolve fixed review threads, update the PR body/reconciliation record when needed, and refresh the watcher head SHA.
- If the watcher reports `attention` with `head_mismatch: true`, stop, fetch the remote head, inspect the delta, and do not force-push over another actor's commit.
- If checks or exact-head review are pending, update this session's state file with the current snapshot and end the model turn without polling again.
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
# CODEX_WAKE_COMMAND="'/home/<you>/.local/bin/atlas-codex-wake-run' --watcher-id '<session-id>' --repo-dir '<absolute repo or worktree path>'"
# Absolute paths only: the bridge shlex.splits this and never uses a shell.
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

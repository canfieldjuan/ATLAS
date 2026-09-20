# PR-Codex-Thread-Resume-Wake

## Why this slice exists

The operator asked for a Codex/local builder that opens a PR, costs nothing
while idle, and wakes when a new review thread appears on that PR. Atlas
already has every stage of that pipeline except the last one, and the last one
has been silently dead since it was written.

The webhook receiver (`~/.local/bin/atlas-pr-webhook-receiver`, user unit
`atlas-pr-webhook-receiver.service`) is enabled and listening on
`127.0.0.1:8765`. It calls `atlas-pr-watch-event`, which snapshots the PR and
calls `scripts/codex_wake_bridge.py --source event`. The bridge builds a wake
prompt and pipes it to the command in `CODEX_WAKE_COMMAND`. That command is an
untracked hand-written local file, `~/.local/bin/atlas-codex-wake-run`, and it
invokes `codex exec --ask-for-approval never`. Current Codex CLI (0.155.1)
rejects that flag outright:

```text
error: unexpected argument '--ask-for-approval' found
Usage: codex exec --cd <DIR> [PROMPT]
```

`~/.local/state/atlas-pr-watchers/codex-wake.log` shows that error on every
wake attempt it has ever recorded. No Codex wake has ever succeeded.

Fixing the flag alone would still leave the expensive shape. `codex exec` with
a bare prompt starts a **fresh thread** each wake, so the woken agent has no
memory of the PR arc it is supposed to be continuing, and the operator pays the
full cold-start context every time. Measured on this machine, a fresh Codex
turn costs about 29k input tokens before it does any work. Thread-resumed
turns re-send history too, but the repeat is served from cache
(`cached_input_tokens`), and the agent actually remembers the arc.

This slice replaces the dead ad-hoc local script with a repo-owned,
installed, tested runner that resumes one persistent Codex thread per watcher.
`docs/long_running_session_watcher_handoff.md` already forbids exactly the
thing that is installed today: "Do not recreate a watcher from ad hoc local
source."

### Problem-derived contract

- Root cause: the Codex wake path terminates in an untracked local script that
  passes a CLI flag the installed Codex no longer accepts, and that starts a
  fresh thread per wake instead of resuming the arc's thread. Because the
  script is not repo-owned, no installer, drift check, test, or CI gate could
  observe that it was broken.
- Correct fix must touch/change: a repo-owned wake runner that (a) invokes only
  flags the installed `codex exec` / `codex exec resume` accept, (b) persists
  the thread id from the first wake and resumes it on later wakes, (c) is
  installed and drift-checked by `scripts/install_codex_wake_bridge.py` like
  every other watcher component, and (d) has tests that fail if the invoked
  argv shape regresses. The handoff doc must name the runner as the bridge's
  wake command.
- Must not change: the bridge's wake classification and attention-only
  semantics (`scripts/codex_wake_bridge.py`), the watcher's read-only posture
  (`scripts/pr_watcher.py`), the readiness proof schema, the webhook receiver,
  the 30-minute scheduled cadence, or anything that could grant merge
  authority to watcher infrastructure. No product code is in scope.

## Scope (this PR)

Ownership lane: dev-workflow/codex-wake-resume
Slice phase: Workflow/process

1. Add `scripts/codex_wake_run.py`: reads the wake prompt on stdin, resumes a
   persisted Codex thread for that watcher id when one exists, otherwise starts
   one and records its id, and serializes concurrent wakes behind a lock.
2. Install and drift-check that runner through
   `scripts/install_codex_wake_bridge.py`, so a future broken runner is caught
   by `--check` instead of by silence in a log.
3. Add tests that pin the invoked argv shape for both the fresh and resume
   paths, the thread-id round trip, and the concurrent-wake behavior.
4. Point `docs/long_running_session_watcher_handoff.md` at the installed runner
   as the `CODEX_WAKE_COMMAND`, replacing the fresh-session example.

### Review Contract

- Acceptance criteria:
  - The fresh path invokes `codex exec --json -` and no flag outside the set
    accepted by `codex exec` on 0.155.1 -- settled by
    `tests/test_codex_wake_run.py::test_fresh_wake_argv_shape`.
  - The resume path invokes `codex exec resume <thread-id> --json -` and passes
    neither `-C/--cd` nor `-s/--sandbox`, because `codex exec resume` accepts
    neither -- settled by
    `tests/test_codex_wake_run.py::test_resume_wake_argv_shape` and
    `::test_resume_never_passes_cd_or_sandbox_flags`.
  - Working directory reaches Codex as the subprocess cwd on both paths, not as
    a flag -- settled by
    `tests/test_codex_wake_run.py::test_repo_dir_is_passed_as_cwd_on_both_paths`.
  - Sandbox reaches Codex as `-c sandbox_mode=<value>` on both paths and
    defaults to `workspace-write`, not full access -- settled by
    `tests/test_codex_wake_run.py::test_sandbox_default_is_workspace_write`.
  - A `thread.started` event on the first wake is persisted, and the next wake
    resumes that exact id -- settled by
    `tests/test_codex_wake_run.py::test_thread_id_round_trip`.
  - A malformed or foreign-shaped stored thread id is ignored and the runner
    starts fresh rather than passing attacker-influenced text as argv --
    settled by `tests/test_codex_wake_run.py::test_rejects_malformed_thread_id`.
  - A second wake while one holds the lock exits 0 without invoking Codex --
    settled by `tests/test_codex_wake_run.py::test_concurrent_wake_is_skipped`.
  - The runner contains no PR merge or delete-branch command, so
    `scripts/audit_pr_watcher_safety.py` stays clean -- settled by
    `python scripts/audit_pr_watcher_safety.py` in Verification.
  - `scripts/install_codex_wake_bridge.py --check` reports the runner as a
    tracked, drift-checked file -- settled by
    `tests/test_install_codex_wake_bridge.py::test_check_detects_runner_drift`.
- Reachability proof: entrypoint is
  `atlas-pr-webhook-receiver -> atlas-pr-watch-event -> codex_wake_bridge.py
  --source event -> CODEX_WAKE_COMMAND`. Observable effect is a Codex turn
  recorded against a stable thread id in
  `~/.local/state/atlas-pr-watchers/<id>.codex-thread` plus a usage line in
  `<id>.codex-wake.log`. The runner's `--dry-run` prints the exact argv without
  spending tokens, which is how the operator verifies wiring.
- Affected surfaces: `scripts/codex_wake_run.py` (new),
  `scripts/install_codex_wake_bridge.py`, `tests/test_codex_wake_run.py` (new),
  `tests/test_install_codex_wake_bridge.py`,
  `docs/long_running_session_watcher_handoff.md`.
- Risk areas: argv drift against a future Codex CLI; thread-id file tampering
  feeding argv; concurrent wakes from a burst of review comments; a resumed
  thread outliving the PR it was opened for; accidental widening of sandbox or
  approval posture in watcher-triggered automation.
- Reviewer rules triggered: R1 (plan/contract), R4 (untrusted input to argv),
  R7 (concurrency), R9 (installed-artifact drift), R14 (workflow safety).

### Boundary-change enumeration

The runner admits one externally-influenced value into a subprocess argv: the
stored thread id. That is an admission boundary.

- Boundary path/seam: `_read_thread_id()` in `scripts/codex_wake_run.py`, the
  only path from `<state-dir>/<watcher-id>.codex-thread` into the Codex argv.
- Replaced-path behaviors: previously no thread id existed and the local script
  always started fresh. The new behavior admits a stored id only when it
  matches the canonical Codex session-id shape; every other value, including an
  empty file, whitespace, a flag-shaped string, or a path, falls back to the
  fresh path rather than failing the wake.
- Guard-relevant fields: the thread-id file contents, the watcher id used to
  build that path, and the resolved state directory.
- Caller x input shape: bridge-invoked wake x {absent file, valid UUID,
  malformed text, leading-dash string, oversized file, unreadable file}; each
  shape is covered in `tests/test_codex_wake_run.py`.

### Deployed-config probing

- Deployed/default config values: `CODEX_WAKE_COMMAND` is currently unset in
  all three configs under `~/.config/atlas-pr-watchers/`, so the bridge writes
  handoff files and runs nothing. This slice does not set it; the operator opts
  in per watcher.
- Explicit value probe: `--sandbox read-only` and `--sandbox workspace-write`
  both reach Codex as `-c sandbox_mode=<value>`.
- Absent value probe: with `--sandbox` omitted the runner sends
  `workspace-write`, never full access.
- Default-session/default-context probe: with no `.codex-thread` file present
  the runner takes the fresh path and creates one.
- Side-effect ordering: the lock is acquired before Codex is invoked, and the
  thread id is written only after `thread.started` is observed, so a killed
  wake cannot leave a thread id that points at nothing.

### Files touched

- TODO: run `python scripts/sync_pr_plan.py plans/PR-Codex-Thread-Resume-Wake.md` after implementation.

## Mechanism

`codex_wake_run.py` reads the prompt on stdin and looks for
`<state-dir>/<watcher-id>.codex-thread`.

The two Codex subcommands do not share a flag surface, which is the detail the
old script got wrong. Verified against 0.155.1: `codex exec` accepts `-C/--cd`
and `-s/--sandbox`; `codex exec resume` accepts neither. Rather than branch the
argv on that asymmetry, the runner uses only mechanisms that exist on both:
the working directory is passed as the subprocess `cwd`, and the sandbox is
passed as `-c sandbox_mode="<value>"`, which both subcommands accept as a
generic config override.

```text
fresh:   codex exec           --json -c sandbox_mode="..." -   (cwd=repo_dir)
resume:  codex exec resume ID --json -c sandbox_mode="..." -   (cwd=repo_dir)
```

`--json` makes stdout a JSONL event stream. The runner scans it for
`thread.started` to capture `thread_id`, and for `turn.completed` to record the
turn's usage. Events are echoed to `<state-dir>/<watcher-id>.codex-wake.log`
with a timestamp, so the operator can see cost per wake instead of inferring it.

A non-blocking `flock` on `<state-dir>/<watcher-id>.codex-wake.lock` serializes
wakes. Review bursts post several comments within seconds, and the webhook
receiver fires per delivery; without the lock, one burst would start several
Codex turns against the same thread. A wake that cannot take the lock logs and
exits 0, because a wake already in flight will observe the same PR state.

## Intentional

- The sandbox default is `workspace-write`, not the `danger-full-access` the
  dead local script used. The bridge prompt is built from PR metadata and
  review text, which the handoff doc already classifies as untrusted input.
  Watcher-triggered automation should not run unsandboxed on that input by
  default; an operator who wants more passes `--sandbox` explicitly.
- No approval flag is passed at all. The old script's `--ask-for-approval never`
  is what broke it, and `approval_policy = "never"` is already set in the
  operator's `~/.codex/config.toml`, so the non-interactive path needs nothing.
  Passing approval posture from a watcher script would also re-hide that
  decision in local state.
- The thread id is validated against the Codex session-id shape before it
  reaches argv. It is a plain file in a state directory, so treating it as
  trusted would let anything that can write there inject argv into a
  `danger-`capable process.
- One persistent thread per watcher id, not per PR. A watcher config is already
  one-per-session-per-PR, so watcher id is the correct grain and requires no new
  identifier.
- `--dry-run` prints argv and exits without invoking Codex, so wiring can be
  verified for free. This matters specifically because the failure this slice
  fixes was invisible for months.

## Deferred

Parking predicate: this slice parks hardening of the wake transport itself
(delivery guarantees, retry, backpressure) and anything touching the scheduled
merge path.

- Thread lifecycle on merge. A merged PR leaves its `.codex-thread` file
  behind, and the next PR reusing that watcher id would resume a stale arc.
  Teardown belongs with the existing post-merge teardown step in AGENTS 3c.1,
  not here. Unlocked by a follow-up slice that extends teardown.
- Context growth over a long arc. A thread resumed across many wakes grows
  until Codex auto-compacts. A per-thread turn or token ceiling that forks a
  fresh thread would bound it; needs a measured threshold first.
- Enabling the timers. `atlas-pr-watch@.timer` and
  `atlas-pr-watch-event@.timer` have no symlinks under
  `~/.config/systemd/user/timers.target.wants/`, so no instance is enabled
  today. Enabling one per owned PR is an operator action, not a repo change.
- The installed `~/.local/bin/atlas-pr-watch` predates the readiness-proof
  schema and reports no `readiness` block. Re-running the installer refreshes
  it; that is an operator step recorded in Verification, not a code change.

Parked hardening: none.

## Verification

- Pending before push: `pytest tests/test_codex_wake_run.py
  tests/test_install_codex_wake_bridge.py tests/test_codex_wake_bridge.py
  tests/test_audit_pr_watcher_safety.py -q`; `python
  scripts/audit_pr_watcher_safety.py`; `bash scripts/check_ascii_python.sh`;
  `python scripts/codex_wake_run.py --watcher-id probe --repo-dir . --dry-run`
  against both a fresh and a seeded thread-id file.

## Estimated diff size

| File | LOC |
|---|---:|
| **Total** | **0** |

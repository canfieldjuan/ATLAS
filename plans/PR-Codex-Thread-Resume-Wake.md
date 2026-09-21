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

**Over the 400 LOC budget, deliberately.** The diff is 3681 diff lines, of
which 1901 are tests and 625 are this plan. Runtime code is 1083 lines
across three files.
The reason this slice is test-heavy rather than divisible is the defect itself:
the thing being replaced failed silently for months because no test pinned the
argv it invoked. Shipping the runner without the argv-shape, malformed-input,
concurrency, and installer-drift tests would reproduce the exact failure mode
this slice exists to close. Splitting runner and tests across two PRs would
leave a window where the same silence is possible.

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
Max files: 10

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
  - The runner contains no PR merge or delete-branch command, so
    `scripts/audit_pr_watcher_safety.py` stays clean -- settled by
    `python scripts/audit_pr_watcher_safety.py` in Verification.
  - `scripts/install_codex_wake_bridge.py --check` reports the runner as a
    tracked, drift-checked file -- settled by
    `tests/test_install_codex_wake_bridge.py::test_check_detects_runner_drift`
    and `::test_check_reports_a_missing_runner`.
  - Each wake records the agent's final message, so a wake that runs while the
    operator is away is auditable afterwards -- settled by
    `tests/test_codex_wake_run.py::test_agent_message_is_recorded_for_the_absent_operator`
    and `::test_long_agent_message_is_truncated_in_the_log_but_kept_in_full`.
  - The safety audit scans both the repo source and the INSTALLED runner, and
    fails on a merge command in either -- settled by
    `tests/test_audit_pr_watcher_safety.py::test_fails_on_codex_wake_runner_with_merge_command`
    and `::test_fails_on_installed_wake_runner_with_merge_command`.
  - The thread id is read inside the wake lock, so two wakes cannot both start
    a thread -- settled by
    `tests/test_codex_wake_run.py::test_thread_id_is_read_inside_the_lock`.
  - Only a CONFIRMED missing session quarantines the stored id, meaning the
    canonical diagnostic naming that exact thread, so an unrelated "not found"
    from another subsystem cannot discard a valid arc -- settled by
    `tests/test_codex_wake_run.py::test_confirmed_missing_session_quarantines_the_id`,
    `::test_next_wake_after_quarantine_starts_fresh`,
    `::test_an_unrelated_not_found_does_not_quarantine`, and
    `::test_reports_missing_session_requires_the_phrase_and_the_id`.
  - A resume comes back as the thread it asked for, or the turn fails and the
    stored id survives, so an arc cannot be silently redirected -- settled by
    `tests/test_codex_wake_run.py::test_a_resume_reporting_a_different_thread_is_refused`
    with the fresh-thread side
    `::test_a_fresh_turn_still_records_whatever_thread_it_started`.
  - The other side of that boundary: a transient pre-attach failure (network,
    auth, config, empty stderr) KEEPS the id, because a wrong quarantine
    permanently loses the arc -- settled by
    `tests/test_codex_wake_run.py::test_transient_resume_failure_keeps_the_id`
    and `::test_failed_resume_that_did_attach_keeps_the_id`.
  - The thread id is persisted the moment `thread.started` is consumed, not at
    end of turn, so a turn killed mid-flight still records the session Codex
    already created -- settled by
    `tests/test_codex_wake_run.py::test_thread_id_persists_when_the_turn_is_killed_mid_flight`.
  - The documented `CODEX_WAKE_COMMAND` is a runnable argv. The bridge
    `shlex.split`s it and never uses a shell, so the doc uses absolute paths --
    settled by `docs/long_running_session_watcher_handoff.md:127-135`.
  - EVERY wake-command example in the doc names the runner, not a bare
    `codex exec`, so an operator following any setup section gets thread
    persistence -- settled by `grep -n CODEX_WAKE_COMMAND
    docs/long_running_session_watcher_handoff.md` returning only runner
    invocations at lines 129, 330, and 478.
  - A wake that arrives while another holds the lock waits for it and then
    runs its own prompt, so no wake is dropped and there is no queue to strand
    -- settled by
    `tests/test_codex_wake_run.py::test_a_blocked_wake_waits_and_then_runs_its_own_turn`.
  - A wake that never gets the lock exits non-zero instead of reporting a
    success it did not perform -- settled by
    `tests/test_codex_wake_run.py::test_lock_timeout_reports_failure_rather_than_dropping_the_wake`.
  - There is no skip path at all, so no file, stamp, or clock change can
    suppress a wake -- settled by
    `tests/test_codex_wake_run.py::test_every_wake_that_gets_the_lock_runs_its_prompt`
    and `::test_a_backward_clock_cannot_suppress_a_wake`.
  - A post-turn audit-write failure does not overturn a completed turn, so a
    caller cannot be driven to repeat real side effects -- settled by
    `tests/test_codex_wake_run.py::test_an_unwritable_audit_path_does_not_fail_a_completed_turn`.
  - The thread id is written durably enough to survive a host restart, not just
    process death: the staged file and its directory entry are both flushed --
    settled by
    `tests/test_codex_wake_run.py::test_atomic_write_flushes_the_file_and_its_directory`.
  - The Codex child does not outlive a killed runner, so an orphan cannot keep
    editing the checkout while a later wake takes the freed lock -- settled by
    `tests/test_codex_wake_run.py::test_codex_child_is_asked_to_die_with_the_runner`
    and `::test_die_with_parent_sets_the_parent_death_signal`, and by a
    before/after experiment recorded in Verification.
  - A runner that dies in the window between fork and setting the death signal
    still does not leak a Codex process, because the hook refuses to exec once
    it has been reparented -- settled by
    `tests/test_codex_wake_run.py::test_hook_refuses_to_exec_when_already_reparented`
    and the live-parent side `::test_hook_execs_normally_when_the_parent_is_still_alive`.
  - Descendants Codex starts do not survive a killed runner either, because the
    turn runs under a supervisor that leads its own process group and takes the
    group down -- settled by
    `tests/test_codex_wake_run.py::test_supervisor_is_interposed_between_the_runner_and_codex`,
    `::test_supervisor_runs_its_command_and_returns_its_exit_code`, and a
    before/after kill experiment recorded in Verification.
  - A turn whose thread id could not be stored does not happen at all, so no
    work is done that nothing can resume -- settled by
    `tests/test_codex_wake_run.py::test_unusable_thread_path_refuses_before_launching_codex`,
    `::test_thread_path_problem_names_a_directory`, and the usable side
    `::test_thread_path_problem_accepts_a_usable_path`.
  - A persistence failure that only appears mid-turn ends in a controlled exit
    rather than a traceback out of running work -- settled by
    `tests/test_codex_wake_run.py::test_a_persist_failure_mid_turn_is_a_controlled_outcome`.
  - A turn that finishes normally but left a background process running has
    that process stopped before the supervisor returns -- settled by
    `tests/test_codex_wake_run.py::test_a_background_process_does_not_outlive_the_lock`,
    with the clean case `::test_a_clean_turn_is_not_slowed_by_the_drain`.
  - When the drain cannot stop something, the turn exits
    `EXIT_TURN_NOT_CONTAINED` instead of reporting success, because the lock is
    released on process exit regardless and a silent leak would read as a clean
    turn -- settled by
    `tests/test_codex_wake_run.py::test_drain_reports_how_many_it_could_not_stop`.
  - A wrong-conversation resume is stopped at the event that names it, before
    it can act -- settled by
    `tests/test_codex_wake_run.py::test_a_wrong_thread_turn_is_stopped_before_it_can_act`.
  - The missing-thread diagnostic is matched as one record, so output about a
    different thread cannot quarantine this one -- settled by
    `tests/test_codex_wake_run.py::test_a_split_diagnostic_does_not_quarantine`.
  - The drain's process scan reports what it could not read, so a scan that
    saw almost nothing is distinguishable from an empty group -- settled by
    `tests/test_codex_wake_run.py::test_scan_reports_what_it_could_not_read`.
  - That scan finds a process whose name contains the `/proc` stat delimiter,
    so a member cannot hide from the drain -- settled by
    `tests/test_codex_wake_run.py::test_pgid_of_parses_every_comm_shape` and
    `::test_scan_finds_a_child_whose_name_contains_the_delimiter`, which runs a
    real binary named `worker) hidden` and fails against the old parser.
  - The supervisor's termination handlers exist before Codex is spawned, so a
    runner that dies in between cannot leave Codex running without the lock --
    settled by
    `tests/test_codex_wake_run.py::test_supervisor_handlers_are_installed_before_the_spawn`
    and a before/after kill experiment recorded in Verification.
  - Containment is claimed only for what process groups can actually deliver,
    so nothing asserts a guarantee a descendant can opt out of -- settled by
    `tests/test_codex_wake_run.py::test_drain_does_not_claim_to_cover_escaped_descendants`
    and the Deferred entry pointing at issue #2526.
  - An unusable state directory exits with a diagnostic rather than a
    traceback, even though it is touched before the log and lock exist --
    settled by
    `tests/test_codex_wake_run.py::test_an_unusable_state_directory_exits_cleanly`.
  - A thread path that is a FIFO, a directory, or a link to one is refused
    without ever being opened, so a wake cannot block holding the lock --
    settled by
    `tests/test_codex_wake_run.py::test_a_special_thread_file_is_rejected_without_being_opened`
    and `::test_a_symlinked_thread_file_is_judged_on_its_own`, which hang
    against the pre-fix code and pass in under a second after it.
  - A host without the kernel's parent-death signal logs that orphan protection
    is not in force rather than skipping it silently -- settled by
    `tests/test_codex_wake_run.py::test_parent_death_support_is_probed_in_the_parent`.
  - The documented wake command survives a repo or worktree path containing
    whitespace, because the bridge shlex-splits the value and the outer quotes
    do not protect individual arguments -- settled by
    `tests/test_codex_wake_end_to_end.py`, whose paths contain spaces and which
    fails against the unquoted form.
  - The real-entrypoint smoke actually runs in CI, rather than being skipped by
    a marker -- settled by `tests/test_codex_wake_end_to_end.py` carrying no
    pytest marker, and by its enrollment in
    `.github/workflows/codex_wake_bridge_checks.yml` both as a path filter and
    in the explicit test list.
  - Codex exiting 0 without naming a thread is treated as a protocol failure,
    not success -- settled by
    `tests/test_codex_wake_run.py::test_zero_exit_without_a_thread_event_is_a_protocol_failure`
    and `::test_zero_exit_with_a_malformed_thread_id_is_a_protocol_failure`.
  - The real entrypoint works end to end: the bridge reads a watcher config,
    splits the documented `CODEX_WAKE_COMMAND`, invokes the INSTALLED runner,
    and a thread plus wake record appear; a second event resumes that thread --
    settled by `tests/test_codex_wake_end_to_end.py`, which fakes only the
    Codex binary.
- Reachability proof: entrypoint is
  `atlas-pr-webhook-receiver -> atlas-pr-watch-event -> codex_wake_bridge.py
  --source event -> CODEX_WAKE_COMMAND`. Observable effect is a Codex turn
  recorded against a stable thread id in
  `~/.local/state/atlas-pr-watchers/<id>.codex-thread` plus a usage line in
  `<id>.codex-wake.log`. The runner's `--dry-run` prints the exact argv without
  spending tokens, which is how the operator verifies wiring.
- Affected surfaces: `scripts/codex_wake_run.py` (new),
  `scripts/install_codex_wake_bridge.py`, `scripts/audit_pr_watcher_safety.py`
  (one line: the runner joins `REPO_WATCHER_SOURCES` so the merge-authority
  scan covers it), `tests/test_codex_wake_run.py` (new),
  `tests/test_install_codex_wake_bridge.py`,
  `tests/test_audit_pr_watcher_safety.py`,
  `docs/long_running_session_watcher_handoff.md`.
- Risk areas: argv drift against a future Codex CLI; thread-id file tampering
  feeding argv; concurrent wakes from a burst of review comments; a resumed
  thread outliving the PR it was opened for; accidental widening of sandbox or
  approval posture in watcher-triggered automation.
- Reviewer rules triggered: R1 (plan/contract), R2 and R10 (this diff edits
  `scripts/audit_pr_watcher_safety.py`, a gate predicate: it widens
  `REPO_WATCHER_SOURCES` so the merge-authority scan covers the new runner, and
  the widening is proved to detect by
  `tests/test_audit_pr_watcher_safety.py::test_fails_on_codex_wake_runner_with_merge_command`
  rather than only to pass), R4 (untrusted stored thread id reaching argv),
  R7 (concurrent wakes serialized by a lock), R9 (installed-artifact drift),
  R14 (workflow safety).

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

- `plans/PR-Codex-Thread-Resume-Wake.md` (new) -- this plan.
- `scripts/codex_wake_run.py` (new) -- the runner.
- `scripts/install_codex_wake_bridge.py` -- install and drift-check it.
- `scripts/audit_pr_watcher_safety.py` -- one line, scan scope.
- `tests/test_codex_wake_run.py` (new) -- argv shape, thread round trip,
  malformed-id admission, lock waiting and supersede, protocol failures,
  wake record.
- `tests/test_codex_wake_end_to_end.py` (new) -- the bridge through the
  installed runner, faking only the Codex binary.
- `.github/workflows/codex_wake_bridge_checks.yml` -- run the two new test
  files, and trigger on the runner and its tests.
- `tests/test_install_codex_wake_bridge.py` -- runner install and drift.
- `tests/test_audit_pr_watcher_safety.py` -- runner merge-authority scan.
- `docs/long_running_session_watcher_handoff.md` -- the wake command operators
  configure, and why it is the runner.

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
`thread.started` to capture `thread_id`, for `turn.completed` to record the
turn's usage, and for the final agent message. The thread id is written the
moment that event is consumed, because Codex has already created the session by
then. Events are echoed to `<state-dir>/<watcher-id>.codex-wake.log` with a
timestamp, so the operator can see cost and outcome per wake.

**Concurrency.** An `flock` on `<state-dir>/<watcher-id>.codex-wake.lock`
serializes wakes. Review bursts post several comments within seconds and the
webhook receiver fires per delivery, so several wakes can overlap. A wake that
cannot take the lock immediately **waits** for it, up to `LOCK_WAIT_SECONDS`,
and then runs its own prompt; it is not dropped and nothing is handed to
another process. A wake that never gets the lock exits `EXIT_LOCK_TIMEOUT`
rather than reporting a success it did not perform. There is no skip path and
no prompt queue, so no state can suppress a wake.

**Turn ownership.** Codex is started under a supervisor
(`--supervise <runner pid> -- <codex argv>`), which calls `setsid` to lead its
own process group and inherits the lock descriptor. Three properties follow.
The kernel's parent-death signal stops the turn when the runner dies; the
supervisor then kills its whole process group, so a shell or test runner Codex
started underneath does not survive; and because the lock lives on the inherited
open file description, it is not released while that group still holds it. The
supervisor also compares its parent against the pid captured before the fork and
exits rather than starting Codex if it has already been orphaned.

**State before effects.** The thread path is checked for usability before Codex
is launched, because a turn whose id cannot be stored does real work that
nothing can resume. If it fails anyway mid-turn, the turn is allowed to finish
and the runner reports `EXIT_STATE_UNUSABLE` instead of raising, and post-turn
diagnostic writes never change a completed turn's outcome.

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
- Non-JSON stdout lines are counted and reported once per turn rather than
  skipped silently. Codex can interleave plain text into the event stream, and
  skipping it quietly would make a garbled stream look identical to a quiet
  one -- the same class of invisibility this slice exists to fix.
- `build_argv` re-validates the sandbox value even though argparse already
  constrains the CLI. The function is importable and the value is interpolated
  straight into argv, so the guard belongs at the interpolation site.
- Opening the wake log and lock is guarded: a background wake whose state
  directory is unwritable exits 2 with a message rather than a traceback nobody
  is present to read.
- A blocked wake waits for the lock and then runs its own prompt. There is no
  prompt queue. Three review rounds were spent closing windows in a file-based
  handoff (drop-on-contention, then a read-then-unlink claim race, then the
  window between the holder's final queue check and its release), which is the
  AGENTS 3k.2 signal to replace the mechanism rather than patch it again. A
  queue whose producer and consumer synchronize through a second file cannot be
  made strand-free by adding checks; waiting removes the second file, so the
  lock is the only shared state and the invariant is one line: every wake that
  acquires the lock runs the prompt it was given.
- **Burst coalescing is removed, deliberately.** It was never part of this
  slice's requirement: the operator asked for a wake that costs nothing while
  idle, and serializing on the lock already delivers that. It was an
  optimization added in response to a review finding, and it went on to produce
  five consecutive rounds of correctness findings -- dropped events on
  contention, a read-then-unlink claim race, the window between a final queue
  check and the unlock, a start stamp treated as proof of a successor, and
  finally wall-clock regression reversing wake order. Each fix was real and
  each introduced the next question, which is the AGENTS 3k.2 signal to delete
  the mechanism rather than harden it again. Ordering wakes correctly needs a
  monotonic, reboot-aware, same-host clock, which is more machinery than the
  optimization is worth here. Removing it leaves no skip path, so no file,
  stamp, or clock change can suppress a wake. The cost is stated in Deferred.
- The Codex child is bound to the runner's lifetime with the kernel's
  parent-death signal rather than a signal handler, because the runner can be
  SIGKILLed and a handler cannot run then. The probe for that facility happens
  in the parent: the pre-exec hook runs after fork with no safe way to report,
  so a missing facility would be silent exactly where it matters.
- The flag alone is not enough, so the hook also checks its parent. Setting
  the death signal happens after fork, and a runner that dies in between has
  already caused the kernel to reparent the child; setting the flag then
  delivers nothing, because it is not retroactive. The hook compares against
  the pid captured before the fork and exits rather than starting Codex into
  an orphan. That check is not conditional on the kernel facility, because
  being orphaned is worth refusing either way.
- Codex runs under a supervisor rather than being spawned directly. The
  parent-death signal reaches exactly one pid, so a shell or test runner Codex
  starts underneath outlives it; passing the lock descriptor does not close
  that either, because an intermediate process that closes inherited
  descriptors breaks the chain. The supervisor leads its own process group,
  holds the lock descriptor, and takes the group down when the runner dies, so
  nothing from a turn outlives the lock that covered it. If it cannot create
  its own group it says so and falls back to stopping the immediate child,
  rather than aiming a group kill at the wrong target.
- The supervisor installs its termination handlers before it spawns anything.
  Doing it afterwards leaves a window in which the parent-death signal takes
  its default action, killing only the supervisor and leaving Codex running
  without the lock descriptor. The handler is written to work before a child
  exists, because everything the supervisor spawns joins its process group at
  fork, so the group kill covers a child spawned moments earlier.
- The group is drained on the normal path too, not only when the runner dies.
  A turn's own process exiting does not mean the turn is over: Codex can start
  a background command that outlives it, and returning then would release the
  lock while that command is still editing the checkout. Leftovers get SIGTERM,
  a short grace period, then SIGKILL, and are reaped before the supervisor
  returns. A turn that left nothing behind pays nothing for this.
- Unusable state is rejected before launch, not discovered afterwards. A turn
  whose thread id cannot be stored still edits files, pushes, and comments, and
  nothing can resume it, so every retry repeats that work. The check runs ahead
  of the turn; the residual mid-turn case ends in a controlled exit code.
- Post-turn writes are diagnostics and never change the turn's outcome. By the
  time the agent message is recorded, Codex may already have edited files,
  pushed, or commented. Failing the wake because that copy could not be written
  would make the caller retry and repeat those side effects, so the failure is
  logged and the turn still reports success.
- The lock cannot be held past process exit, so a drain that cannot stop
  everything reports the leak and exits non-zero rather than returning a clean
  turn. Holding ownership until the group is genuinely empty would mean
  blocking indefinitely on something this runner cannot kill; making the leak
  loud is the honest alternative, and issue #2526 carries the containment that
  would remove the case.
- Quarantine requires the canonical diagnostic AND the id it names, captured
  from the same match. An earlier
  version also accepted bare phrases like "session not found", which an
  unrelated message such as "MCP server session not found" satisfies, throwing
  away a valid thread over a failure that had nothing to do with it. That is
  the precise harm quarantining exists to prevent, so the matcher is now the
  verified phrase plus the thread being resumed.
- A resume must report the thread it asked for. Writing a different id would
  redirect the arc and every later wake with it, so a mismatch keeps the stored
  id and fails the turn. A fresh turn, with nothing stored, still records
  whatever thread it starts.
- Quarantine is keyed to a confirmed missing-session signature on stderr, not
  to a nonzero exit. Codex reports it as
  `no rollout found for thread id <uuid> (code -32600)`, which is why stderr is
  captured to a file and inspected rather than streamed straight to the log. An
  unrecognized pre-attach failure keeps the id and logs loudly, because a wrong
  quarantine permanently loses the arc while keeping the id costs only a retry.
  The quarantined id is moved aside rather than deleted so it stays inspectable.

## Deferred

Parking predicate: this slice parks hardening of the wake transport itself
(delivery guarantees, retry, backpressure) and anything touching the scheduled
merge path.

- Thread lifecycle on merge. A merged PR leaves its `.codex-thread` file
  behind, and the next PR reusing that watcher id would resume a stale arc.
  Teardown belongs with the existing post-merge teardown step in AGENTS 3c.1,
  not here. Unlocked by a follow-up slice that extends teardown.
- Context growth over a long arc. A thread resumed across many wakes grows.
  Measured on this machine across three real wakes on one thread, `input_tokens`
  per turn ran 29,605 then 59,237 then 88,909, with `cached_input_tokens`
  reaching 29,602 by the third. The growth is roughly linear in turns, and cache
  offsets the repeat but does not remove it from the plan's usage. A per-thread
  turn or token ceiling that forks a fresh thread would bound it; the numbers
  above are the starting point for choosing that threshold. Tracked as
  follow-up, not fixed here, because picking a ceiling needs data from a real
  multi-day arc rather than a three-turn probe.
- **Containment a descendant cannot leave (issue #2526).** A descendant that
  calls `setsid` leaves the supervisor's process group, and nothing built from
  process groups can stop it, because leaving is the descendant's choice. The
  lock is then released while that process still runs. This is not fixed here:
  the real fix is a cgroup or a transient systemd scope, `systemd-run --user`
  did not respond on the dev host during this slice, and doing it properly
  likely replaces the hand-rolled supervisor rather than extending it. What
  this slice does instead is stop claiming the guarantee, in the code, the plan
  and the handoff doc, so the limit is visible rather than assumed.
- Extracting turn lifetime into a reviewed component. Owning a Codex turn's
  lifetime has needed a supervisor, a process group, a parent-death signal, a
  reparent check, handler ordering around the spawn, and a drain on both the
  normal and abnormal paths. Five review rounds found a real window in that
  mechanism each time. Every fix stands on its own and the measured behavior is
  now correct, but this is a lot of process-lifetime surface for a wake runner
  to carry, and a purpose-built supervisor (or a systemd scope, which gets the
  same property from the kernel) would be a better home for it. Not attempted
  here because replacing the mechanism mid-review would discard the evidence
  already gathered for the current one.
- Burst coalescing, if measurement shows it is worth it. Without it, N review
  comments arriving together cost N serialized turns instead of one. Idle cost
  is unchanged at zero, which is the property this slice was asked for. A
  correct coalescer needs a monotonic, reboot-aware ordering source rather than
  wall time, and a skip predicate based on completed work rather than intent;
  both were attempted here and are recorded above as the reason it was removed.
  The wake log records per-turn token usage, so the real cost of a burst can be
  measured before rebuilding it.
- Enabling the timers. `atlas-pr-watch@.timer` and
  `atlas-pr-watch-event@.timer` have no symlinks under
  `~/.config/systemd/user/timers.target.wants/`, so no instance is enabled
  today. Enabling one per owned PR is an operator action, not a repo change.
- The installed `~/.local/bin/atlas-pr-watch` predates the readiness-proof
  schema and reports no `readiness` block. Re-running the installer refreshes
  it; that is an operator step recorded in Verification, not a code change.

Parked hardening: none.

## Verification

Run against the final tree on this branch, not carried over from an earlier
round:

- `pytest tests/test_codex_wake_run.py tests/test_codex_wake_end_to_end.py
  tests/test_install_codex_wake_bridge.py tests/test_codex_wake_bridge.py
  tests/test_audit_pr_watcher_safety.py tests/test_pr_watcher.py
  tests/test_report_pr_watcher_state.py -q` -- **312 passed**.
- `python scripts/audit_pr_watcher_safety.py` -- exit 0, "watcher
  docs/config/source grant no merge authority".
- `bash scripts/check_ascii_python.sh` -- exit 0.
- `python scripts/maturity_sweep.py scripts --tests-root tests --baseline
  tests/maturity_sweep/baseline_scripts.json --min-score 8 --sensitive-glob
  'scripts/**'` -- exit 0, no new brittleness above baseline.
- End-to-end against the **real** `codex-cli 0.155.1`: a fresh wake stored
  codeword `GUNWALE-9903` and a resumed wake recalled it, through the
  supervisor and with the current identity and quarantine rules in place.

Probes that shaped specific fixes, each reproduced before the change and
re-run after it:

- Safety-audit detection: appending `gh pr merge --delete-branch` to the runner
  made `audit_pr_watcher_safety.py` exit 1 and name the file, so the scan
  detects rather than merely passing. Reverted; the durable form is
  `test_fails_on_codex_wake_runner_with_merge_command`.
- Orphaned child: SIGKILLing the runner left Codex alive with the lock free
  without the parent-death hook, and killed it with the hook.
- Escaped descendant: a Codex that backgrounds a sleeper left it running and
  the lock free before the group drain, and nothing after it.
- Blocking thread path: with the path as a FIFO, the pre-fix code hung even on
  `--dry-run`; the fixed code returns `EXIT_STATE_UNUSABLE` immediately.
- Wrong-conversation resume: a resume of A reporting B used to overwrite the
  stored id, return 0, and let that turn write a side-effect marker first. It
  now stops the turn at the event and keeps A.
- `python scripts/install_codex_wake_bridge.py --check` reports
  `content drift: ~/.local/bin/atlas-codex-wake-run`, which is the broken local
  script this slice replaces becoming visible to tooling for the first time.

## Estimated diff size

| File | +/- |
|---|---:|
| `tests/test_codex_wake_run.py` | +1528 |
| `scripts/codex_wake_run.py` | +1059 |
| `plans/PR-Codex-Thread-Resume-Wake.md` | +625 |
| `tests/test_codex_wake_end_to_end.py` | +238 |
| `tests/test_audit_pr_watcher_safety.py` | +89 |
| `docs/long_running_session_watcher_handoff.md` | +63 / -4 |
| `tests/test_install_codex_wake_bridge.py` | +46 |
| `scripts/install_codex_wake_bridge.py` | +13 |
| `scripts/audit_pr_watcher_safety.py` | +11 |
| `.github/workflows/codex_wake_bridge_checks.yml` | +5 |
| **Total** | **3681** |

Over the 400 LOC soft cap. Runtime code is 508 lines; the remainder is tests (794), this plan (381), and docs (51). The growth over the first push is six Codex review findings and their regression tests, all fixed rather than waived.

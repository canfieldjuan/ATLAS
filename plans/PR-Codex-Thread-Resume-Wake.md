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

**Over the 400 LOC budget, deliberately.** The diff is 1738 diff lines, of
which 794 are tests and 381 are this plan. Runtime code is 508 lines
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
Max files: 8

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
  - Only a CONFIRMED missing session quarantines the stored id, so the next
    wake recovers to a fresh thread instead of retrying a dead resume forever --
    settled by
    `tests/test_codex_wake_run.py::test_confirmed_missing_session_quarantines_the_id`
    and `::test_next_wake_after_quarantine_starts_fresh`.
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
  - A wake that arrives while another holds the lock is queued rather than
    dropped, and the lock holder drains it before exiting -- settled by
    `tests/test_codex_wake_run.py::test_concurrent_wake_queues_instead_of_dropping`
    and `::test_lock_holder_drains_a_prompt_queued_mid_turn`.
  - Queue draining is capped so a review burst cannot chain Codex turns without
    bound, and the event that does not fit is preserved -- settled by
    `tests/test_codex_wake_run.py::test_coalescing_is_capped_and_leaves_the_remainder`.
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
  malformed-id admission, concurrency, wake record.
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
- A wake blocked on the lock queues its prompt instead of dropping it, and
  the lock holder drains the queue before releasing. The earlier "the in-flight
  wake observes the same PR state" assumption was wrong: the running turn may
  already have taken its snapshot, so a review posted after that point would
  have been invisible until some unrelated later event.
- Queue draining is capped at `MAX_COALESCED_TURNS`. Newest-wins coalescing
  already collapses a burst, but an unbounded drain loop would be a token sink
  of exactly the kind this slice exists to avoid. A prompt that does not fit is
  left queued for the next wake rather than discarded.
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
- Enabling the timers. `atlas-pr-watch@.timer` and
  `atlas-pr-watch-event@.timer` have no symlinks under
  `~/.config/systemd/user/timers.target.wants/`, so no instance is enabled
  today. Enabling one per owned PR is an operator action, not a repo change.
- The installed `~/.local/bin/atlas-pr-watch` predates the readiness-proof
  schema and reports no `readiness` block. Re-running the installer refreshes
  it; that is an operator step recorded in Verification, not a code change.

Parked hardening: none.

## Verification

Run on this branch before push:

- `pytest tests/test_codex_wake_run.py tests/test_install_codex_wake_bridge.py
  tests/test_codex_wake_bridge.py tests/test_audit_pr_watcher_safety.py
  tests/test_pr_watcher.py tests/test_report_pr_watcher_state.py -q`
  -- **234 passed**.
- `python scripts/audit_pr_watcher_safety.py` -- exit 0, "watcher
  docs/config/source grant no merge authority".
- Negative probe of that audit: appending `gh pr merge --delete-branch` to the
  runner made it exit 1 and name `scripts/codex_wake_run.py`, so the scan
  detects rather than merely passing. Reverted; the durable version of that
  probe is `test_fails_on_codex_wake_runner_with_merge_command`.
- `bash scripts/check_ascii_python.sh` -- exit 0.
- `--dry-run` against the real CLI printed
  `codex exec --json -c sandbox_mode="read-only" -` for the fresh path and
  `codex exec resume <id> --json -c sandbox_mode="read-only" -` for the seeded
  path, with cwd set to the repo dir on both.
- End-to-end against the **real** `codex-cli 0.155.1`, three wakes on one
  watcher id: wake 1 started a thread and recorded
  `01a0bfdf-c300-7e40-aff1-2c4c1949a2a9`; wakes 2 and 3 resumed that same id;
  wake 3 was asked for a codeword stored in wake 1 and answered `ANVIL-3392`.
  That is the continuity proof -- the argv the runner builds is accepted by the
  installed binary, and the resumed thread carries the arc.
- `python scripts/install_codex_wake_bridge.py --check` now reports
  `content drift: ~/.local/bin/atlas-codex-wake-run`, which is the broken local
  script this slice replaces becoming visible to tooling for the first time.

## Estimated diff size

| File | +/- |
|---|---:|
| `tests/test_codex_wake_run.py` | +659 |
| `scripts/codex_wake_run.py` | +484 |
| `plans/PR-Codex-Thread-Resume-Wake.md` | +381 |
| `tests/test_audit_pr_watcher_safety.py` | +89 |
| `docs/long_running_session_watcher_handoff.md` | +51 / -4 |
| `tests/test_install_codex_wake_bridge.py` | +46 |
| `scripts/install_codex_wake_bridge.py` | +13 |
| `scripts/audit_pr_watcher_safety.py` | +11 |
| **Total** | **1738** |

Over the 400 LOC soft cap. Runtime code is 508 lines; the remainder is tests (794), this plan (381), and docs (51). The growth over the first push is six Codex review findings and their regression tests, all fixed rather than waived.

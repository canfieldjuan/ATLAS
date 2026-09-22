# PR-Codex-Wake-Profile-Wiring

Implementation of the contract accepted in
`plans/PR-Codex-Wake-Profile-Isolation.md`, merged as PR #2532. That document
is the contract; this one records how it was built and where reality forced a
correction to it.

## Why this slice exists

The accepted contract specifies that a wake turn should run under a profile the
operator chooses rather than the one the runner happens to inherit. Nothing
implements it yet, so every wake still inherits the interactive profile: 75,685
tokens on a first wake and 87,509 added per wake after, against a conversation
of about 90 tokens.

### Problem-derived contract

**Root cause.** `scripts/codex_wake_run.py` builds a child process for `codex`
without deciding that child's profile. Codex resolves its injected context from
`$CODEX_HOME` and from `$HOME/.agents/skills`, and neither is a per-invocation
flag, so the child inherits whatever the runner was launched with.

**What a correct fix must touch.** The construction of the child environment,
and nothing else in the launch path. Two variables, because one root lives
under each.

**What must not change.** Argv shape for `exec` and `exec resume`; the wake
lock; turn containment; the thread-id, agent-message and usage receipts; every
exit code; and the behavior of a deployment that configures no profile.

**One correction to the accepted contract.** Its reachability criterion says an
isolated first wake must show `input_tokens` near the 27,495 row rather than the
75,685 row. Building it showed that criterion is unsound. A real wake's
`input_tokens` is dominated by the work the agent does, not by the profile: the
live chain run below cost 116,309 because the agent actually investigated the
PR (2,145 output tokens, 1,178 reasoning, several tool calls), while the
earlier interactive-profile run through the same chain cost 76,228 only because
Codex was blocked at its first command by this host's AppArmor restriction and
did nothing. Token count is therefore a proxy for effort, not for profile. The
corrected criterion is the one this slice proves: the effective-profile receipt
appears in the watcher's own log, which is deterministic and in-chain, and the
scaffolding saving stays settled by the controlled matched pair that holds the
prompt and the work constant.

## Scope (this PR)

Ownership lane: dev-workflow/codex-wake-resume
Slice phase: Workflow/process
Max files: 5

1. Add `--codex-home` and `--agent-home` to the runner and apply them to the
   child environment.
2. Log the effective profile on every launching turn (contract I5).
3. Record which profile owns a stored thread id, and start fresh rather than
   attempting a cross-profile resume (contract I8).
4. Tests for all four argument combinations, the receipt, partial isolation,
   fail-closed launch, and the cross-profile thread case.
5. Document the arguments and the measured profile layout in the handoff.

### Files touched

- `scripts/codex_wake_run.py`
- `tests/test_codex_wake_run.py`
- `docs/long_running_session_watcher_handoff.md`
- `plans/PR-Codex-Wake-Profile-Wiring.md` (new)

### Review Contract

- All four argument combinations reach the child, and no profile means an
  untouched environment -- settled by
  `tests/test_codex_wake_run.py::test_every_profile_argument_combination_reaches_the_child`
  (parametrized over the closed four-state inventory) and
  `::test_no_profile_leaves_the_child_environment_untouched`, which asserts
  `child_environment` returns `None` so `Popen` is called exactly as today.
- The receipt names the effective profile whatever its origin -- settled by
  `::test_the_receipt_names_the_effective_profile_however_it_was_set`, which
  sets `CODEX_HOME` in the environment, passes no argument, and requires
  `(inherited)` in the log. Negative-probed: deleting the log line makes it fail.
- Partial isolation does not read as full isolation -- settled by
  `::test_partial_isolation_is_recorded_as_partial`, which asserts the warning
  is present with one root and absent with both.
- A misconfigured profile is launched into rather than pre-validated -- settled
  by `::test_a_missing_profile_is_launched_into_and_fails_closed`, which
  asserts Codex was invoked, received the nonexistent path unchanged, and that
  its exit code survived.
- Switching profiles and back resumes the original arc -- settled by
  `::test_switching_profiles_and_back_resumes_the_original_arc`, which runs
  three wakes across two profiles and asserts the map holds both arcs.
  Negative-probed: keeping only the newest entry makes it fail.
- A legacy id resumes and its ownership is backfilled on attach -- settled by
  `::test_a_legacy_thread_id_is_adopted_and_recorded_on_attach`, so a later
  profile change cannot quarantine it.
- The id and its owner are one record -- settled by
  `::test_the_thread_id_and_its_owner_are_one_record`.
- The map is read through the bounded safe path -- settled by
  `::test_the_thread_map_is_read_through_the_bounded_safe_path`, which plants a
  FIFO and requires the read to return inside a deadline.
- An unusable map starts fresh rather than guessing -- settled by
  `::test_an_unusable_thread_map_starts_fresh_rather_than_guessing`.
- `CODEX_HOME` is derived from an isolated `HOME` rather than reported unset --
  settled by `::test_codex_home_is_derived_from_an_isolated_home`.
  Negative-probed: removing the derivation makes it fail.
- Nothing PR #2525 settled regressed -- settled by the full suite at 379 passed.

**Reachability proof.** The configured chain was run, not the runner directly:
`atlas-pr-watch-and-wake wake-profile-proof`, whose watcher config carries both
profile arguments inside `CODEX_WAKE_COMMAND`. The wake log at
`~/.local/state/atlas-pr-watchers/wake-profile-proof.codex-wake.log` records
`profile codex_home=/home/juan-canfield/.codex-wake (argument)
home=/home/juan-canfield/.codex-wake-home (argument)`, a recorded thread id, a
sibling `.codex-thread.profile` naming the owning profile, and a `usage=` line.

**Risk areas.** A deployment with no profile configured behaving differently;
an existing watcher losing its arc on upgrade; the profile marker disagreeing
with the stored id after a partial write.

**Reviewer rules triggered.** R1 (plan/contract), R2 (runtime change), R8
(fail-closed boundary), R13 (receipts).

## Mechanism

`resolve_profile` turns the two optional arguments plus the current environment
into a `WakeProfile` that carries both the arguments and the **effective**
values the child will see, each tagged `argument`, `inherited` or `unset`.
`child_environment` returns `None` when nothing is configured, so `Popen` is
called with no `env=` at all and an unconfigured deployment is untouched.

Thread ids are kept in one atomically replaced JSON map, `{codex_home:
thread_id}`, beside the existing single-id file. Each profile therefore keeps
its own arc: switching away and back resumes the original thread rather than
starting a third. Pairing the id with its owner inside one document also makes
a mismatched pair unrepresentable, where an id file plus a separate owner file
could disagree if the process died between the two writes.

`thread_id_for_profile` resolves what this profile may resume. A map entry for
the effective `CODEX_HOME` is resumed. No map at all means the watcher predates
this change, so its single stored id is adopted by the profile running now and
written into the map on attach, which is the backfill that stops a later
profile change from attempting a cross-profile resume. A map that names other
profiles but not this one starts fresh, and the other arcs stay resumable. An
unreadable or malformed map starts fresh and says so, because unknown ownership
is not the same as no ownership.

The map is read through the same descriptor-based path as the thread id, with
`O_NONBLOCK`, `O_NOFOLLOW`, a regular-file check and a bounded read, so a
planted FIFO cannot hang a wake that holds the lock.

`resolve_profile` also accounts for Codex having no unset `CODEX_HOME`: it
defaults to `$HOME/.codex`, and `HOME` is a value this runner may itself be
changing. Verified against the real CLI, which created `<home>/.codex` and
authenticated against it. So an isolated `HOME` isolates both roots, and that
combination is not reported as partial.

## Intentional

- No pre-launch profile validation, because the contract forbids it and
  reproduction showed Codex already fails closed and free.
- One map rather than an id file plus an owner file. Two independent durable
  writes can disagree after a crash between them, leaving a valid new id paired
  with the previous owner, after which the next wake treats its own thread as
  foreign. A single document makes that state unrepresentable.
- The single-id file is kept as a mirror, because it is the documented,
  human-readable pointer to the arc a watcher is on and its strict one-id
  format is what keeps a malformed value out of argv.
- An absent map is treated as "this watcher predates the map" and its id is
  adopted, not discarded. The opposite would make every existing watcher start
  a fresh thread on upgrade, which is the exact harm the invariant prevents.
- An isolated `HOME` alone is not reported as partial isolation, because Codex
  derives its home from `HOME` and both roots move together.
- `child_environment` returns `None` rather than a copy of `os.environ`.
  Copying would be equivalent in practice but would change the call shape, and
  the contract's I2 is about being byte-identical.

## Deferred

Parking predicate: this slice parks profile provisioning and the operator
rollout, not correctness of the mechanism.

- Installer support for creating and drift-checking a wake profile.
- Enabling the profile on the live watcher configs, which is an operator action.
- Amending PR #2532's reachability criterion on main. This plan records the
  correction; editing the merged contract is a docs-only follow-up.
- A per-thread turn or token ceiling, carried forward from #2525 and #2532.

Parked hardening: none.

## Verification

- Command: `pytest tests/test_codex_wake_run.py tests/test_codex_wake_end_to_end.py -q` - Result: 161 passed - Environment: local
- Command: `pytest tests/test_codex_wake_run.py -q -k "another_profile or effective_profile"` with the cross-profile check disabled - Result: fail - Environment: local
- Command: `pytest tests/test_codex_wake_run.py -q -k "effective_profile or partial_isolation"` with the receipt line removed - Result: fail - Environment: local
- Command: `~/.local/bin/atlas-pr-watch-and-wake wake-profile-proof` - Result: pass - Environment: local
- Command: `bash scripts/check_ascii_python.sh` - Result: pass - Environment: local

The two `fail` results are negative probes: each new regression test was shown
to fail with its fix removed and pass with it restored.

## Estimated diff size

| File | +/- |
|---|---:|
| `tests/test_codex_wake_run.py` | +321 |
| `scripts/codex_wake_run.py` | +316 |
| `plans/PR-Codex-Wake-Profile-Wiring.md` | +211 |
| `docs/long_running_session_watcher_handoff.md` | +32 |
| **Total** | **887** |

Diff-budget override: 887 lines against a 400-line soft cap. Runtime change is 316 lines; the other two thirds are the regression tests the contract names and
the plan. Splitting tests from the behavior they pin would leave a window where
a cross-profile resume silently discards an arc with nothing to catch it, which
is the defect this slice exists to prevent.

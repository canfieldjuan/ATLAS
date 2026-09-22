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
- The map destination is preflighted like the thread path -- settled by
  `::test_the_thread_map_destination_is_preflighted`, which asserts Codex is
  never launched when the map path is unusable.
- A reset clears both the id file and the map, and removing the id file alone
  does not discard a mapped arc -- settled by
  `::test_resetting_a_watcher_clears_both_the_id_file_and_the_map`. An earlier
  revision treated the id file's absence as a global reset; that could not work
  because the file is shared by every profile, so a quarantine of one profile
  invalidated the rest.
- Quarantining one profile leaves another profile's arc resumable -- settled by
  `::test_quarantining_one_profile_leaves_another_profiles_arc_resumable`.
- Enabling a profile on an existing watcher keeps the legacy arc under the home
  that created it -- settled by
  `::test_enabling_a_profile_keeps_the_legacy_arc_for_its_own_home`, which
  asserts the new profile starts fresh and the legacy id is remembered against
  the baseline home rather than adopted.
- No crash schedule can produce a partial map -- settled by the Execution model
  below, whose invariant holds over every interleaving the surface admits
  rather than over a list of sampled windows.
  `::test_an_emptied_map_does_not_re_adopt_the_quarantined_mirror` remains as
  one worked example of that invariant, not as the proof of it.
- A profile too long to represent is dropped rather than written oversized --
  settled by
  `::test_a_retained_profile_that_cannot_fit_is_dropped_not_written_oversized`.
  Reproduced at 4,091 characters serializing to 8,230 bytes.
- The writer cannot produce a map the reader rejects -- settled by
  `::test_the_thread_map_writer_cannot_outgrow_the_reader`. Reproduced first:
  the old writer produced 10,353 bytes against a reader limit of 8,192, after
  which every remembered arc read back as a size error.
- The map is bounded by size alone, never by a count of profiles, and a
  size-forced eviction is reported -- settled by
  `::test_representable_profiles_are_never_evicted_by_count` and
  `::test_size_forced_eviction_is_reported_not_silent`. Reproduced first: an
  eight-entry cap evicted the first of nine profiles that together serialized
  to 444 bytes against an 8,192-byte limit, so switching back to it started a
  fresh thread.
- A dead session is forgotten in the map, not only the mirror -- settled by
  `::test_a_dead_session_is_forgotten_in_the_map_not_only_the_mirror`, which
  asserts the other profile keeps its arc. Negative-probed.
- An inherited `CODEX_HOME` with an isolated `HOME` is reported partial --
  settled by `::test_an_inherited_codex_home_with_an_isolated_home_is_partial`.
  Negative-probed.
- `CODEX_HOME` is derived from an isolated `HOME` rather than reported unset --
  settled by `::test_codex_home_is_derived_from_an_isolated_home`.
  Negative-probed: removing the derivation makes it fail.
- Nothing PR #2525 settled regressed -- settled by the eight-file wake suite, the command and its 390-pass count recorded in Verification.

**Reachability proof.** The configured chain was run against this head, not the
runner directly and not an earlier installed build:
`atlas-pr-watch-and-wake wake-profile-proof`, twice, with a watcher config
carrying both profile arguments inside `CODEX_WAKE_COMMAND`. The wake log at
`~/.local/state/atlas-pr-watchers/wake-profile-proof.codex-wake.log` records
`profile codex_home=/home/juan-canfield/.codex-wake (argument)
home=/home/juan-canfield/.codex-wake-home (argument)` on both turns, a fresh
turn recording thread `01a0c7c0-6e23-7e60-8398-dbdcec6b6581` and a second turn
resuming that same id. The artifact produced beside the thread id is this head's thread map, a JSON
document named for the watcher, holding
`{"/home/juan-canfield/.codex-wake": "01a0c7c0-..."}`. No profile-marker file
exists, because nothing in this implementation produces one, which is how the
run is known to have exercised this head rather than the earlier runner.

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

`thread_id_for_profile` resolves what this profile may resume, and the map is
the only authority for it. A map entry for the effective `CODEX_HOME` is
resumed. A map that exists but does not name this profile starts fresh, and the
other arcs stay resumable. An unreadable or malformed map starts fresh and says
so, because unknown ownership is not the same as no ownership.

With no map file at all the watcher predates this change, and its single stored
id was created before any profile argument existed, so it belongs to the home
that would be in effect with no arguments. It is resumed only when the wake is
running under that same home, and it is written into the map against that home
on the first attach. Adopting it for a newly enabled profile would resume a
rollout that home does not hold, after which the missing-session path renames
the only id file and strands the original arc.

The id file is a mirror for humans, never a per-profile signal. An earlier
revision used its absence as a reset, which cannot work: it is shared by every
profile, so the rename a quarantine performs would have invalidated arcs that
were still resumable. Resetting a watcher removes both files, and both the
handoff doc and the runner's own failure diagnostic say so.

The map is read through the same descriptor-based path as the thread id, with
`O_NONBLOCK`, `O_NOFOLLOW`, a regular-file check and a bounded read, so a
planted FIFO cannot hang a wake that holds the lock.

`resolve_profile` also accounts for Codex having no unset `CODEX_HOME`: it
defaults to `$HOME/.codex`, and `HOME` is a value this runner may itself be
changing. Verified against the real CLI, which created `<home>/.codex` and
authenticated against it. So an isolated `HOME` isolates both roots, and that
combination is not reported as partial.

### Execution model

Required because this slice adds durable state. The previous revisions of this
plan listed crash schedules to handle, which is the enumeration the rule
rejects: each review round found a schedule the list had omitted.

**Surface.** Two files in one directory on one local POSIX filesystem, written
and read by a single process that already holds the per-watcher `flock`. No
network, no leases, no clocks, no partitions, no retry or redelivery. The modes
this surface admits are therefore: process or host death at any instruction,
and out-of-band mutation of either file by an operator or another tool.

**Invariant, over every interleaving the surface admits.** The thread map is
the only input to the resume decision, and it is only ever replaced whole,
through write, fsync, `os.replace`, fsync of the directory. A reader therefore
observes exactly one of two states: the complete previous map, or the complete
new one. There is no third, partial state to reason about, so crash timing
cannot produce one. Both observable states are correct: the previous map means
the most recent attach is forgotten and that profile starts fresh, which costs
a thread and never resumes a wrong one; the new map means it is remembered.

The single-id file is derived output and never an input to that decision, with
one stated exception: when no map file exists at all, it is read once to
migrate a watcher created before this change, attributed to the home that would
be in effect with no profile arguments. After the first attach a map always
exists, so that exception cannot be reached again.

Those two sentences are the whole model. Every question of the form "what if it
dies between X and Y" resolves to "does a map file exist, and if so it is
complete", which needs no schedule to be enumerated.

**Assumptions, stated rather than omitted.**

- `os.replace` is atomic and the two fsyncs order data before the rename and
  the rename before it is durable. This holds on Linux within one filesystem,
  which is why both files live in the one state directory. It is not assumed
  across filesystems.
- The per-watcher lock means no second writer. If an operator runs the runner
  with the lock bypassed, last-writer-wins applies; the map is still never
  partial, so the failure is a forgotten arc, not a corrupt one.
- An operator who removes only the map returns that watcher to the migration
  path, which re-adopts the id file under the baseline home. This is a
  consequence of the model, not a defect, and the documented reset removes both
  files.
- Nothing here is safe against a filesystem that reorders a renamed entry past
  its own data without honouring fsync. That is assumed not to happen.

**Component rejected.** `sqlite3` is in the standard library and would supply
atomic multi-key updates without a hand-rolled protocol. It is rejected because
the wake state is deliberately operator-readable and operator-editable: the
documented reset is removing files, the handoff doc tells an operator to read
the thread id, and the drift-checking installer inspects plain files. SQLite
would add journal and WAL files to the same state directory that none of that
tooling knows about, and would put a second durability surface inside a slice
whose purpose is profile isolation.

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
  kept, not discarded. The opposite would make every existing watcher start a
  fresh thread on upgrade, which is the exact harm the invariant prevents. It
  is resumed only under the home that created it, and otherwise remembered
  against that home so enabling a profile cannot orphan it.
- No count cap on the map. One was tried and removed on reproduction: it evicted
  representable arcs far under the byte limit. Size is the only bound the reader
  enforces, so it is the only bound the writer applies.
- Reset names two files rather than inferring itself from one. Inferring it
  from the shared id file was tried and reverted: it made a quarantine of one
  profile invalidate every other profile's arc.
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

- Command: `pytest tests/test_codex_wake_run.py tests/test_codex_wake_end_to_end.py -q` - Result: 172 passed - Environment: local
- Command: `pytest tests/test_codex_wake_bridge.py tests/test_codex_wake_run.py tests/test_codex_wake_end_to_end.py tests/test_codex_issue_queue.py tests/test_install_codex_wake_bridge.py tests/test_pr_watcher.py tests/test_report_pr_watcher_state.py tests/test_audit_pr_watcher_safety.py -q` - Result: 390 passed - Environment: local
- Command: `pytest tests/test_codex_wake_run.py -q -k "another_profile or effective_profile"` with the cross-profile check disabled - Result: fail - Environment: local
- Command: `pytest tests/test_codex_wake_run.py -q -k "effective_profile or partial_isolation"` with the receipt line removed - Result: fail - Environment: local
- Command: `~/.local/bin/atlas-pr-watch-and-wake wake-profile-proof` - Result: pass - Environment: local
- Command: `bash scripts/check_ascii_python.sh` - Result: pass - Environment: local

- Command: `pytest tests/test_codex_wake_run.py -q -k "map_destination_is_preflighted or removing_the_thread_id_file or emptied_map_does_not_re_adopt or writer_cannot_outgrow"` against the pre-fix runner - Result: fail - Environment: local

Every regression test in this slice was run against the pre-fix code and shown
to fail there before being accepted as proof. One of them initially failed only
because the old code lacked a helper, which is not a reproduction, so the
writer/reader size mismatch was reproduced directly instead: the old writer
produced 10,353 bytes against a reader limit of 8,192.

The `fail` results above are negative probes: each new regression test was shown
to fail with its fix removed and pass with it restored.

## Estimated diff size

| File | +/- |
|---|---:|
| `tests/test_codex_wake_run.py` | +634 |
| `scripts/codex_wake_run.py` | +444 |
| `plans/PR-Codex-Wake-Profile-Wiring.md` | +343 |
| `docs/long_running_session_watcher_handoff.md` | +49 |
| **Total** | **1481** |

Diff-budget override: 1481 lines against a 400-line soft cap. Runtime change is 444 lines; the other two thirds are the regression tests the contract names and
the plan. Splitting tests from the behavior they pin would leave a window where
a cross-profile resume silently discards an arc with nothing to catch it, which
is the defect this slice exists to prevent.

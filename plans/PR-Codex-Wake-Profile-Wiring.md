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

Diff-budget override: this PR is over the 400-line soft cap; the generated
Estimated diff size section carries the exact figure. The runtime change is
small. Most of the rest is regression tests, each shown to fail on the pre-fix
code where a behavioural failure is possible, and a plan whose execution model,
closure declaration and reconciled findings the repo rules require. Splitting
the tests from the behaviour they pin would leave a window in which a
cross-profile resume can silently discard or misattribute an arc, which is the
defect this slice exists to prevent.

### Problem-derived contract

**Root cause.** Two facts, both reproduced against codex-cli 0.155.1.

1. `scripts/codex_wake_run.py` builds a child process for `codex` without
   deciding that child's profile. Codex resolves its injected context from
   `$CODEX_HOME` and from `$HOME/.agents/skills`, and neither is a
   per-invocation flag, so the child inherits whatever the runner was launched
   with.
2. A thread id is only meaningful inside the `CODEX_HOME` that created it.
   Resuming one under another profile returns `no rollout found for thread id
   ... (code -32600)`, and the runner then quarantines it. So once a watcher can
   run under more than one profile, which file holds a thread id is part of the
   fix, not an unrelated persistence change.

**What a correct fix must touch.** The construction of the child environment,
for fact 1, and the choice of which thread file a wake reads and writes, for
fact 2. Nothing else in the launch path.

**What must not change.** Argv shape for `exec` and `exec resume`; the wake
lock; turn containment; the thread-id, agent-message and usage receipts; every
exit code; and the thread persistence mechanism itself: the single-id file
format, the descriptor-based reader, the atomic writer and the quarantine.

**Compatibility invariant.** The accepted contract listed thread persistence
and resume as must-not-change, and fact 2 was only established afterwards. The
invariant that replaces that prohibition is narrower and checkable: the
persistence mechanism is unchanged, and only the path it operates on is chosen
by that wake's effective Codex home, including a wake given no profile
arguments, whose home is whatever it inherits. A watcher created before this
change keeps its arc: its single `<watcher>.codex-thread` is adopted once into
the inherited home's file, which is the home the pre-profile runner used, and
renamed to `.migrated`.

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

1. Add `--codex-home` and `--agent-home`, admit only non-empty absolute paths,
   and apply them to the child environment.
2. Log the effective profile and the thread file on every launching turn
   (contract I5).
3. Give each effective Codex home its own thread file, including the inherited
   one, and adopt a pre-profile thread file into the inherited home's file
   once (contract I8).
4. Add `--reset-threads`, which removes exactly one watcher's thread files
   under its wake lock, and document it in place of a filename glob.
5. Give atomic-write staging files unpredictable names.
6. Tests for each of the above, and the handoff doc.

### Files touched

- `docs/long_running_session_watcher_handoff.md`
- `plans/PR-Codex-Wake-Profile-Wiring.md`
- `scripts/codex_wake_run.py`
- `tests/test_codex_wake_end_to_end.py`
- `tests/test_codex_wake_run.py`

### Review Contract

- All four argument combinations reach the child, and no profile means an
  untouched environment -- settled by
  `tests/test_codex_wake_run.py::test_every_profile_argument_combination_reaches_the_child`
  and `::test_no_profile_leaves_the_child_environment_untouched`, which asserts
  `child_environment` returns `None` so `Popen` gets no `env=` at all.
- Argument values outside the admitted domain never start a turn -- settled by
  `::test_profile_arguments_outside_the_admitted_domain_never_launch`, over the
  empty string, whitespace, a relative path and a `~` path for both flags. All
  eight cases fail behaviourally on the pre-fix runner, which launched the turn.
  The empty case was reproduced against the real CLI first: Codex treats an
  empty `CODEX_HOME` as unset and uses `$HOME/.codex`, while the runner keyed it
  as `""`, so two different Codex homes shared one thread slot.
- Spellings of one path share one thread file -- settled by
  `::test_trailing_slash_spellings_of_one_home_share_one_thread_file`.
- The receipt names the effective profile whatever its origin -- settled by
  `::test_the_receipt_names_the_effective_profile_however_it_was_set`.
- Partial isolation is recorded as partial, including an inherited
  `CODEX_HOME` with an isolated `HOME` -- settled by
  `::test_partial_isolation_is_recorded_as_partial` and
  `::test_an_inherited_codex_home_with_an_isolated_home_is_partial`.
- `CODEX_HOME` is derived from an isolated `HOME` rather than reported unset --
  settled by `::test_codex_home_is_derived_from_an_isolated_home`.
- A misconfigured profile is launched into rather than pre-validated, and
  records no thread for any profile -- settled by
  `::test_a_missing_profile_is_launched_into_and_fails_closed`.
- The persistence mechanism is unchanged -- settled by every #2525 test, which
  runs with only its fixture path computed through `profile_thread_path`.
- A watcher from before this change keeps its arc -- settled by
  `::test_a_pre_profile_thread_id_is_adopted_once_and_the_old_file_kept`, which
  asserts the pre-profile id resumes, lands in the inherited home's file, and
  the old file is kept as `.migrated`.
- An inherited home change never resumes the other home's arc -- settled by
  `::test_an_inherited_home_change_does_not_resume_the_other_homes_arc`, which
  fails behaviourally on the pre-fix runner with exit 76: the second home
  resumed the first home's id. The migration is also one-shot and crash-safe --
  settled by `::test_a_migrated_thread_is_not_adopted_again_under_another_home`
  and `::test_migration_is_skipped_once_any_home_file_exists`, both of which
  fail with exit 76 on the pre-fix runner.
- A pre-profile file that exists but cannot be inspected is reported rather
  than read as absent -- settled by
  `::test_an_uninspectable_pre_profile_file_is_reported_not_read_as_absent`,
  which fails on the version that swallowed the error.
- `--dry-run` shows a pending migration without performing it -- settled by
  `::test_dry_run_shows_a_pending_migration_without_performing_it`.
- Reset touches exactly one watcher's thread files and waits for its lock --
  settled by `::test_watcher_thread_files_match_exactly_one_watcher`,
  `::test_reset_threads_removes_exactly_this_watchers_thread_files` and
  `::test_reset_threads_waits_for_the_wake_lock`. These fail on the pre-fix
  runner only because the function and flag did not exist; the behavioural
  specimen is the reproduction in which `foo.codex-thread*` matched the files of
  the distinct valid watcher `foo.codex-thread-review`.
- Each profile keeps its own arc and switching back resumes it -- settled by
  `::test_each_profile_keeps_its_own_arc_and_switching_back_resumes`.
- Enabling a profile leaves the pre-profile arc with the inherited home --
  settled by `::test_enabling_a_profile_leaves_the_legacy_arc_resumable`, which
  asserts the new profile starts fresh and the pre-profile id sits in the
  inherited home's file. It fails with exit 76 on the version that adopted the
  pre-profile id into whichever profile ran first.
- A dead session quarantines only its own profile's file -- settled by
  `::test_a_dead_session_quarantines_only_its_own_profile_file`.
- Writers for different profiles cannot lose each other's arc -- settled by
  `::test_writers_for_different_profiles_cannot_lose_each_others_arc`. This one
  cannot fail behaviourally on the pre-fix code: the variable changed was the
  storage model itself, so the old code has no per-profile path to call. Its
  specimen is the reproduction recorded under Execution model.
- The profile's own thread file is preflighted -- settled by
  `::test_the_selected_profile_thread_file_is_preflighted`.
- A leftover staging file cannot fail the post-turn write, and a name planted
  at the old staging path is never followed -- settled by
  `::test_a_stale_staging_file_does_not_fail_the_thread_write` and
  `::test_a_name_planted_at_the_old_staging_path_is_neither_followed_nor_fatal`.
  The first fails with `FileExistsError` on `main`'s pid-named staging.
- Nothing PR #2525 settled regressed -- settled by the eight-file wake suite
  recorded in Verification.

**Closure declaration.** Three inventories drive decisions here.

- **Profile arguments: CLOSED.** The canonical source of membership is the
  argparse definition in `_build_parser`, and membership is derived from it,
  not listed separately: `--codex-home` and `--agent-home`, each given or not,
  which is four combinations. Anything outside that set is an unknown argument
  that argparse rejects with exit 2 before any turn exists. Adding a third
  profile argument changes the parser, and the parametrized combination test
  must grow with it.
- **Profile argument values: CLOSED.** Membership is derived from what Codex
  does with the value: a non-empty absolute path, normalized lexically.
  `profile_directory_argument` is the single admission point. Everything else,
  including empty, whitespace, relative and `~` paths, is rejected with exit 2
  before any turn exists.
- **A watcher's thread files: CLOSED.** Membership is derived from one exact
  pattern in `_watcher_thread_pattern`: the pre-profile file, per-home files
  with a 16-hex-digit digest, and their `.stale` or `.migrated` forms, for
  exactly one watcher id. Any other name, including another watcher's whose id
  merely starts with this one, is outside the set and never touched by a reset.
- **Profile states: OPEN, by design.** Whether a directory is a usable profile
  is decided by Codex and changes with Codex. Outside-set behaviour is defined:
  the runner launches into any admitted path unchanged and the existing receipts
  record whatever Codex does. There is no fallback to select, and no
  state-specific branch may be added.

**Reachability proof.** The configured chain was exercised, not the runner
directly: `atlas-pr-watch-and-wake`, whose `CODEX_WAKE_COMMAND` carries both
profile arguments. Two real turns on the round-nine head, `b28cd309a`, logged
`profile codex_home=/home/juan-canfield/.codex-wake (argument)
home=/home/juan-canfield/.codex-wake-home (argument)
thread_file=wake-profile-proof.codex-thread.7cbbb7246829a74e`; the first
recorded thread `01a0cad7-b821-7080-ae09-3fda1018b339` and the second resumed
it. This round changed the lock helper, migration and the dry-run path, so the
chain was run again on this head with `--dry-run` added to the configured
command, which spends no tokens: the real bridge ran it, and the runner
selected the same per-home file and reported `argv=codex exec resume
01a0cad7-b821-7080-ae09-3fda1018b339 ...`, the thread the real turns recorded.
The resume itself on this head is covered by
`tests/test_codex_wake_end_to_end.py`, which drives the real bridge against a
fake Codex.

**Risk areas.** A deployment with no profile behaving differently; an existing
watcher losing its arc on upgrade; one profile's arc reaching another's turn.

**Reviewer rules triggered.** R1 (plan/contract), R2 (runtime change), R8
(fail-closed boundary and durable state), R13 (receipts).

## Mechanism

`profile_directory_argument` admits a profile argument only if it is a
non-empty absolute path, and normalizes it lexically. `resolve_profile` turns
the arguments plus the current environment into a `WakeProfile` carrying the
**effective** values the child will see, each tagged `argument`, `inherited`,
`derived from HOME` or `unset`. Codex has no unset `CODEX_HOME`: it defaults to
`$HOME/.codex`, verified against the real CLI, so an unset one is derived from
the child's `HOME`. `child_environment` returns `None` when nothing is
configured, so `Popen` is called exactly as before.

`profile_thread_path` chooses the thread file: `<watcher>.codex-thread.<digest>`,
where the digest is the first 16 hex characters of the SHA-256 of the
effective `CODEX_HOME`. Every wake is keyed this way, including one given no
profile arguments, whose effective home is the one it inherits. From there on,
`run_one_turn` is the #2525 code: the same reader, the same atomic writer, the
same quarantine, applied to that one file.

Under the wake lock, before the turn, `migrate_legacy_thread` adopts a
pre-profile `<watcher>.codex-thread` into the inherited home's file, the home
the pre-profile runner used, and renames the old file to `.migrated`. It does
so only while the watcher has no per-home file at all, and it reports rather
than skips a pre-profile file it cannot inspect. `--reset-threads` takes the
same lock and removes exactly the files `watcher_thread_files` enumerates.

`_atomic_write` stages through `tempfile.mkstemp` in the target directory,
which opens with `O_CREAT|O_EXCL|O_NOFOLLOW` at mode 0600 and picks an
unpredictable name.

### Execution model

**Surface.** One single-valued file per effective Codex home per watcher, in one directory
on one local POSIX filesystem. Wakes of a watcher hold its `flock`, so no two
wakes of one watcher run at once. An operator or another tool may create,
replace or remove any of these files at any time. There is no network, lease,
clock or retry.

**Invariant, over every interleaving that surface admits.** A wake reads and
writes only its own home's file, and each file holds one id and is replaced
whole. Therefore no interleaving can make a wake resume another profile's arc,
because it never reads another profile's file, and no interleaving can make a
wake erase another profile's arc, because no write touches more than one file
and no write is a read-modify-write. The only races left are on a single value,
and they resolve to that value's last whole write.

The one exception is migration, which reads the pre-profile file and writes the
inherited home's file. It runs under the wake lock, at most once: it only runs
while the watcher has no per-home file, and it renames the pre-profile file
away afterwards. A crash between the write and the rename leaves a per-home file
in place, so the next wake does not adopt the old file again under whatever
home it runs in.

**Assumptions, stated rather than omitted.**

- `os.replace` is atomic within one filesystem, and the fsyncs order data
  before the rename and the rename before it is durable. All thread files live
  in one state directory.
- The digest is 64 bits. Two effective homes colliding in it would share a
  file; at the handful of profiles a watcher sees, that is negligible, and it is
  assumed not to happen.
- A symlinked alias of a home is a different string, so it reads as a
  different profile. That errs toward a fresh thread, never a wrong one.
- The pre-profile runner used whatever home it inherited, so adopting its id
  into the inherited home is right unless that inherited environment changed
  while the old runner was still in use. That risk existed before this change,
  and migration takes it once.
- Resets are serialized with wakes by `--reset-threads`, which takes the wake
  lock. Removing files some other way while a wake is in flight loses to that
  wake, which writes its own home's file when it records its thread.

**Specimen and isolation.** The shared-map design this replaces was reproduced
losing an update: with profile B recorded out of band between profile A's read
and write, the final map held only A. Holding that interleaving fixed and
changing one variable, from one shared map to one file per profile, made the
failure disappear, which identified the shared read-modify-write document as the
cause.

**Component rejected.** `sqlite3` would provide atomic multi-key updates, but
it is not needed once no update touches more than one key. The wake state is
also deliberately readable and resettable with ordinary file commands, which
the handoff doc relies on.

## Intentional

- **One file per profile rather than a shared map.** Earlier revisions of this
  PR kept every profile's id in one JSON map. Over rounds two to nine, review
  found one defect after another in it: a lost update between read and write, a
  presence check racing its own read, a size bound that evicted valid arcs, an
  eviction order that emptied the map, a migration path that misattributed a
  shared mirror file, and staging collisions. Each was real and each was
  reproduced. They were symptoms of one choice, a multi-key document with a
  read-modify-write, and they are removed by construction rather than fixed
  one at a time. The reconciliation ledger keeps those entries as history; the
  behaviour each protected is now pinned against per-profile files.
- **Every wake is keyed by its effective home, including the inherited one.**
  An earlier revision kept the pre-profile file for any wake without profile
  arguments. That keyed the file by "whatever the environment says now", so a
  watcher whose inherited HOME or CODEX_HOME changed between wakes resumed one
  home's arc inside another and quarantined it. The pre-profile file is now a
  one-time migration source instead.
- **Migration goes to the inherited home, not to whichever profile runs first.**
  The pre-profile runner took no profile arguments, so its id belongs to the
  home a wake inherits. Adopting it into a newly enabled `--codex-home` would
  resume an id that home does not hold.
- **A reset command rather than a filename glob.** Watcher ids may contain dots
  and hyphens, so `<id>.codex-thread*` matches other valid watchers' files. The
  command enumerates an exact pattern and takes the wake lock, which also closes
  the teardown race a manual deletion has.
- **A digest rather than the path in the filename.** A `CODEX_HOME` can be a
  long absolute path that is not safe to embed in a filename. The receipt line
  names the file in use, so the operator can always see which one a wake used.
- **Argument domain enforced at parse time, profile validity left to Codex.**
  Rejecting an empty or relative argument is not profile validation; it refuses
  a value that would name a different directory than the operator meant. What
  happens inside an admitted directory stays Codex's decision.
- **Lexical normalization only.** `realpath` would touch the filesystem and
  could race; lexical normalization cannot, and its only miss, a symlinked
  alias, errs toward a fresh thread.
- **`child_environment` returns `None`** rather than a copy of `os.environ`,
  so an unconfigured deployment launches exactly as before.
- **Unpredictable staging names.** #2525's pid-derived staging name stopped a
  planted symlink by making any leftover fatal, which failed a finished turn
  after Codex had acted whenever a killed wake's pid was reused. `mkstemp` keeps
  the protection and removes the failure. Files left by a hard kill are rare,
  tiny and inert, and are not swept.

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

- Command: `pytest tests/test_codex_wake_run.py tests/test_codex_wake_end_to_end.py -q` - Result: 181 passed - Environment: local
- Command: `pytest tests/test_codex_wake_bridge.py tests/test_codex_wake_run.py tests/test_codex_wake_end_to_end.py tests/test_codex_issue_queue.py tests/test_install_codex_wake_bridge.py tests/test_pr_watcher.py tests/test_report_pr_watcher_state.py tests/test_audit_pr_watcher_safety.py -q` - Result: 399 passed - Environment: local
- Command: `pytest tests/test_codex_wake_run.py -q -k "outside_the_admitted_domain"` against the pre-fix runner - Result: fail - Environment: local
- Command: `pytest tests/test_codex_wake_run.py -q -k "stale_staging_file_does_not_fail_the_thread_write"` against the runner on `main` - Result: fail - Environment: local
- Command: `pytest tests/test_codex_wake_run.py -q -k "inherited_home_change or migrated_thread_is_not_adopted or migration_is_skipped"` against the pre-fix runner - Result: fail - Environment: local
- Command: `pytest tests/test_codex_wake_run.py -q -k "enabling_a_profile_leaves"` against the version that adopted into the current profile - Result: fail - Environment: local
- Command: `pytest tests/test_codex_wake_run.py -q -k "uninspectable"` against the version that swallowed the lstat error - Result: fail - Environment: local
- Command: `~/.local/bin/atlas-pr-watch-and-wake wake-profile-proof`, run twice on `b28cd309a` - Result: pass - Environment: local
- Command: `~/.local/bin/atlas-pr-watch-and-wake wake-profile-proof-dry`, the same configured command with `--dry-run`, on this head - Result: pass - Environment: local
- Command: `python scripts/maturity_sweep.py scripts --tests-root tests --baseline tests/maturity_sweep/baseline_scripts.json --min-score 8 --sensitive-glob 'scripts/**'` - Result: pass - Environment: local
- Command: `bash scripts/check_ascii_python.sh` - Result: pass - Environment: local
- Command: `python scripts/sync_pr_plan.py plans/PR-Codex-Wake-Profile-Wiring.md origin/main --check` - Result: pass - Environment: local

The `fail` results are the regression tests shown failing on the code before
their fix.

## Estimated diff size

| File | LOC |
|---|---:|
| `docs/long_running_session_watcher_handoff.md` | 67 |
| `plans/PR-Codex-Wake-Profile-Wiring.md` | 389 |
| `scripts/codex_wake_run.py` | 520 |
| `tests/test_codex_wake_end_to_end.py` | 11 |
| `tests/test_codex_wake_run.py` | 727 |
| **Total** | **1714** |

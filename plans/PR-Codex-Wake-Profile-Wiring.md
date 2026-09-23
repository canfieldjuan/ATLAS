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
by that wake's canonical effective Codex home, including a wake given no profile
arguments, whose home is whatever it inherits. The single `<watcher>.codex-thread`
file of a watcher created before this change is never read: that watcher's next
wake starts one fresh thread, and `--reset-threads` removes the old file. This
is a deliberate narrowing, recorded under Intentional.

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
3. Give each canonical effective Codex home its own thread file, including the
   inherited one (contract I8). The pre-profile file is never read.
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
  `::test_trailing_slash_spellings_of_one_home_share_one_thread_file` and
  `::test_one_home_spelled_two_ways_resumes_one_arc`, which fails behaviourally
  on 029e00624: an inherited `CODEX_HOME=<home>/` and a later `--codex-home
  <home>/` selected different files, so the second wake started fresh.
- A retargeted profile link never shares an arc, and switching it back
  resumes the first -- settled by
  `::test_a_retargeted_profile_link_does_not_share_an_arc`, which fails with
  exit 76 on 3eaeace38: the wake through the link retargeted to B resumed A's
  id.
- A queued wake resolves its profile only after taking the lock -- settled by
  `::test_a_queued_wake_resolves_its_profile_after_taking_the_lock`, which runs
  a real second process blocked on the real lock and retargets the link while
  it waits. On f630a21ee it exits 76: the profile had been resolved before the
  wait, so it keyed A's file and resumed A's id under B. `run_wake` now takes
  the raw arguments rather than a resolved profile, so no caller can resolve
  outside the lock.
- `--dry-run` and `--reset-threads` cannot be combined -- settled by
  `::test_dry_run_and_reset_threads_cannot_be_combined`; on f630a21ee the pair
  removed the thread file and exited 0.
- `..` is never collapsed into another home, and the child sees the argument
  as written -- settled by
  `::test_dotdot_after_a_symlink_is_not_collapsed_into_another_home`, which
  fails behaviourally on 639de1256, where the argument reached the child
  rewritten and `<root>/link/../profile` was keyed as `<root>/profile`. The
  real CLI was checked first: with `link` pointing into another tree, `codex
  doctor --json` reported that tree's `profile`.
- Every path spelling keys the directory Codex opens -- settled by
  `::test_path_spellings_key_the_directory_codex_opens`, one row per member of
  the path-spelling inventory below, plus
  `::test_a_relative_repo_dir_is_keyed_against_the_process_directory` and
  `::test_a_non_utf8_profile_path_keys_logs_and_resumes`. On 8cbb93e12 the
  doubled-leading-slash row fails, the relative `--repo-dir` wake resumes the
  other directory's id (exit 76), and the non-UTF-8 wake raises
  `UnicodeEncodeError`.
- A relative inherited home is keyed where Codex resolves it -- settled by
  `::test_a_relative_inherited_home_is_keyed_per_repository`, which fails
  behaviourally on 029e00624: the wake in the second repository ran `exec resume`
  with the id recorded under the first repository's home. Codex's own resolution
  was checked first: `CODEX_HOME=rel codex doctor --json` reported `<cwd>/rel`
  from two different directories.
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
- **Every thread-state transition, in one table** -- settled by
  `::test_thread_state_table`. It runs every operation (wake, dead session,
  dry run, reset) on every starting state the file model admits (this home's
  file absent, valid or unusable; another home's file absent or present; the
  pre-profile file absent or present; a neighbouring watcher whose id begins
  with this one's file name always present), and asserts the whole state
  directory afterwards, not only the file under test. A wake resumes only from
  its own home's file and changes only that file; a dead session moves only that
  file to `.stale`; a dry run changes nothing; a reset removes every file of this
  watcher and nothing of the neighbour. On 029e00624 the cell with only the
  pre-profile file present fails: the migration rewrote a file that was not the
  waking home's.
- An inherited home change never resumes the other home's arc -- settled by
  `::test_an_inherited_home_change_does_not_resume_the_other_homes_arc`.
- Reset touches exactly one watcher's thread files, waits for its lock and
  flushes the directory -- settled by
  `::test_watcher_thread_files_match_exactly_one_watcher`,
  `::test_reset_threads_removes_exactly_this_watchers_thread_files`,
  `::test_reset_threads_waits_for_the_wake_lock` and
  `::test_reset_threads_flushes_the_state_directory`, the last of which fails on
  029e00624, which issued no directory fsync after removing files.
- Each profile keeps its own arc and switching back resumes it -- settled by
  `::test_each_profile_keeps_its_own_arc_and_switching_back_resumes`.
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
  does with the value: a non-empty absolute path, passed to the child exactly
  as written. `profile_directory_argument` is the single admission point. Everything else,
  including empty, whitespace, relative and `~` paths, is rejected with exit 2
  before any turn exists.
- **Path spellings of one profile directory: CLOSED by construction.** The
  key is not built from the spelling. `_canonical_directory` joins a relative
  value to the child's working directory (made absolute against this
  process's) and then asks the kernel, through `os.path.realpath`, which
  directory it names now. That is the same resolution Codex reports: `codex
  doctor --json` given `CODEX_HOME=<link>` reports the link's target. So every
  spelling difference is settled by the one source that defines it: `.`,
  repeated or doubled leading slashes, a trailing slash, `..` after a link,
  aliases, and a link retargeted between wakes. Encoding is settled separately
  by hashing filesystem bytes. The parametrized spelling test builds a real
  tree with real links and has a row per kind of difference; it is evidence,
  not the membership list.
- **A watcher's thread files: CLOSED.** Membership is derived from one exact
  pattern in `_watcher_thread_pattern`: the pre-profile file, per-home files
  with a 16-hex-digit digest, and the `.stale` form of each, for
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
it. Later rounds changed how the key is derived and removed migration, so the
chain is run again on each head with `--dry-run` added to the configured
command, which spends no tokens: the real bridge runs it, and the runner must
select the same per-home file and report `argv=codex exec resume
01a0cad7-b821-7080-ae09-3fda1018b339 ...`, the thread the real turns recorded.
The configured `--codex-home` and `--agent-home` contain no symlinks and no
spelling differences, so resolving them leaves the digest unchanged, which the
dry run on each head confirms.
The resume itself on this head is covered by
`tests/test_codex_wake_end_to_end.py`, which drives the real bridge against a
fake Codex.

**Risk areas.** A deployment with no profile behaving differently; an existing
watcher losing its arc on upgrade; one profile's arc reaching another's turn.

**Reviewer rules triggered.** R1 (plan/contract), R2 (runtime change), R8
(fail-closed boundary and durable state), R13 (receipts).

## Mechanism

`profile_directory_argument` admits a profile argument only if it is a
non-empty absolute path, and passes it through unchanged. `resolve_profile` turns
the arguments plus the current environment into a `WakeProfile` carrying the
**effective** values the child will see, each tagged `argument`, `inherited`,
`derived from HOME` or `unset`. Each effective value is the directory Codex
will use: joined to the child's working directory, `--repo-dir`, when
relative, then resolved with `os.path.realpath`, as Codex resolves it; a
derived `<home>/.codex` is resolved again, since it may be a link. The digest
is taken over the filesystem bytes, and the receipt shows non-UTF-8 bytes as
backslash escapes. The resolved values feed only the
thread key and the receipt; the child's environment is never rewritten from
them. Codex has no unset `CODEX_HOME`: it defaults to `$HOME/.codex`, verified
against the real CLI, so an unset one is derived from the child's `HOME`. `child_environment` returns `None` when nothing is
configured, so `Popen` is called exactly as before.

`profile_thread_path` chooses the thread file: `<watcher>.codex-thread.<digest>`,
where the digest is the first 16 hex characters of the SHA-256 of the
effective `CODEX_HOME`. Every wake is keyed this way, including one given no
profile arguments, whose effective home is the one it inherits. From there on,
`run_one_turn` is the #2525 code: the same reader, the same atomic writer, the
same quarantine, applied to that one file.

No wake reads or writes the pre-profile `<watcher>.codex-thread`.
`--reset-threads` takes the same lock, removes exactly the files
`watcher_thread_files` enumerates, and then flushes the state directory, as a
wake does after recording an id.

`_atomic_write` stages through `tempfile.mkstemp` in the target directory,
which opens with `O_CREAT|O_EXCL|O_NOFOLLOW` at mode 0600 and picks an
unpredictable name.

### Execution model

**Surface.** One single-valued file per effective Codex home per watcher, in one directory
on one local POSIX filesystem. Every writer of these files is this runner: a
wake recording or quarantining its own home's id, or `--reset-threads`. Every
one of them holds the watcher's `flock` while it writes, so no two run at once.
Editing the state directory by hand while a wake or reset runs is outside the
model, and the handoff doc directs operators to `--reset-threads` instead.
There is no network, lease, clock or retry.

**Invariant, over every interleaving that surface admits.** A wake reads and
writes only its own home's file, and each file holds one id and is replaced
whole. Therefore no interleaving can make a wake resume another profile's arc,
because it never reads another profile's file, and no interleaving can make a
wake erase another profile's arc, because no write touches more than one file
and no write is a read-modify-write. The only races left are on a single value,
and they resolve to that value's last whole write.

A reset lists this watcher's files and removes them while holding the same
lock, so no admitted writer can add a file between the listing and the return,
and its directory flush makes the removals as durable as the write they undo.

**Assumptions, stated rather than omitted.**

- `os.replace` is atomic within one filesystem, and the fsyncs order data
  before the rename and the rename before it is durable. All thread files live
  in one state directory.
- The digest is 64 bits. Two effective homes colliding in it would share a
  file; at the handful of profiles a watcher sees, that is negligible, and it is
  assumed not to happen.
- A profile's path is resolved under the wake lock, immediately before the
  turn. Retargeting a profile link between that resolution and Codex opening
  the profile is outside the model, like hand-editing the state directory
  during a wake; retargeting between wakes is inside it and keys the new
  target's own file.

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
- **Every wake is keyed by its canonical effective home, including the
  inherited one.** An earlier revision kept the pre-profile file for any wake
  without profile arguments. That keyed the file by "whatever the environment
  says now", so a watcher whose inherited HOME or CODEX_HOME changed between
  wakes resumed one home's arc inside another and quarantined it. A later one
  keyed an inherited home by its raw spelling, so `/srv/codex/` inherited and
  `/srv/codex/` given as an argument were two homes, and a relative inherited
  value was one home across repositories that Codex resolves to two.
- **No migration of the pre-profile file.** Earlier revisions adopted the
  old id into the inherited home once. That was the source of most of those
  rounds' findings: which home to adopt into, a crash between adoption and
  rename, an unreadable file poisoning later wakes. What it protected was
  small. When this shipped, the only pre-profile file on the host was a smoke
  test's, no watcher timer was enabled, and the cost of not migrating is one
  fresh thread per existing watcher. Dropping it removes a whole state from the
  table rather than guarding each of its transitions.
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
- **Key by the kernel's resolution, not by text rules.** Rounds eleven to
  thirteen each added a lexical rule for one more way a spelling can differ
  from the directory, and round fourteen found one no text rule can see: a
  link retargeted between wakes. Codex itself keys its profile by the resolved
  path, so the key now asks the kernel the same question. The earlier
  objection to `realpath`, that it touches the filesystem and could race, is
  narrower than the defect it removes: the only race left is retargeting a
  link during that watcher's own wake, which the execution model excludes.
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
- A shared length bound on watcher ids across the runner, the bridge and
  `pr_watcher` (#2534). The per-home suffix moves the point at which an id is
  too long to store from about 225 to about 208 characters; either way the wake
  fails closed before launch, and real ids here are at most 36 characters.

Parked hardening: none.

## Verification

- Command: `pytest tests/test_codex_wake_run.py tests/test_codex_wake_end_to_end.py -q` - Result: 233 passed, 8 skipped - Environment: local
- Command: `pytest tests/test_codex_wake_bridge.py tests/test_codex_wake_run.py tests/test_codex_wake_end_to_end.py tests/test_codex_issue_queue.py tests/test_install_codex_wake_bridge.py tests/test_pr_watcher.py tests/test_report_pr_watcher_state.py tests/test_audit_pr_watcher_safety.py -q` - Result: 451 passed, 8 skipped - Environment: local
- Command: `pytest tests/test_codex_wake_run.py -q -k "outside_the_admitted_domain"` against the pre-fix runner - Result: fail - Environment: local
- Command: `pytest tests/test_codex_wake_run.py -q -k "stale_staging_file_does_not_fail_the_thread_write"` against the runner on `main` - Result: fail - Environment: local
- Command: `pytest tests/test_codex_wake_run.py -q -k "inherited_home_change"` against the runner before per-home keying - Result: fail - Environment: local
- Command: `pytest tests/test_codex_wake_run.py -q -k "state_table or spelled_two_ways or keyed_per_repository or flushes_the_state_directory"` against the runner at 029e00624 - Result: fail - Environment: local
- Command: `pytest tests/test_codex_wake_run.py -q -k "dotdot_after_a_symlink or finds_nothing_names or trailing_slash_spellings"` against the runner at 639de1256 - Result: fail - Environment: local
- Command: `pytest tests/test_codex_wake_run.py -q -k "spellings_key_the_directory or relative_repo_dir or non_utf8_profile"` against the runner at 8cbb93e12 - Result: fail - Environment: local
- Command: `pytest tests/test_codex_wake_run.py -q -k "retargeted_profile_link or spellings_key_the_directory"` against the runner at 3eaeace38 - Result: fail - Environment: local
- Command: `pytest tests/test_codex_wake_run.py -q -k "queued_wake_resolves or cannot_be_combined"` against the runner at f630a21ee - Result: fail - Environment: local
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
| `docs/long_running_session_watcher_handoff.md` | 76 |
| `plans/PR-Codex-Wake-Profile-Wiring.md` | 451 |
| `scripts/codex_wake_run.py` | 508 |
| `tests/test_codex_wake_end_to_end.py` | 11 |
| `tests/test_codex_wake_run.py` | 1062 |
| **Total** | **2108** |

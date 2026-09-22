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
- A thread id from another profile starts fresh and is not quarantined --
  settled by `::test_a_thread_from_another_profile_starts_fresh_without_quarantine`.
  Negative-probed: disabling the check makes it fail with exit 76.
- A thread id with no recorded owner still resumes -- settled by
  `::test_a_thread_with_no_recorded_profile_still_resumes`, so no existing
  watcher loses its arc on upgrade.
- Nothing PR #2525 settled regressed -- settled by the full suite at 157 passed.

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

The thread id gains a sibling marker file holding the `CODEX_HOME` that owns it.
Before a resume, `foreign_thread_reason` compares the marker to the effective
profile; a mismatch logs why and drops the id, so no resume is attempted and the
missing-session quarantine cannot fire. An absent marker counts as a match,
which keeps every watcher that predates this change resuming as before.

## Intentional

- No pre-launch profile validation, because the contract forbids it and
  reproduction showed Codex already fails closed and free.
- The marker is a sibling file rather than a second line in the thread file,
  because the thread file's strict one-id format is what keeps a malformed
  value out of argv.
- An absent marker is treated as a match rather than a mismatch. The opposite
  would make every existing watcher start a fresh thread on upgrade, which is
  the exact harm this invariant exists to prevent.
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

- Command: `pytest tests/test_codex_wake_run.py tests/test_codex_wake_end_to_end.py -q` - Result: 158 passed - Environment: local
- Command: `pytest tests/test_codex_wake_run.py -q -k "another_profile or effective_profile"` with the cross-profile check disabled - Result: fail - Environment: local
- Command: `pytest tests/test_codex_wake_run.py -q -k "effective_profile or partial_isolation"` with the receipt line removed - Result: fail - Environment: local
- Command: `~/.local/bin/atlas-pr-watch-and-wake wake-profile-proof` - Result: pass - Environment: local
- Command: `bash scripts/check_ascii_python.sh` - Result: pass - Environment: local

The two `fail` results are negative probes: each new regression test was shown
to fail with its fix removed and pass with it restored.

## Estimated diff size

| File | +/- |
|---|---:|
| `tests/test_codex_wake_run.py` | +250 |
| `scripts/codex_wake_run.py` | +234 |
| `plans/PR-Codex-Wake-Profile-Wiring.md` | +175 |
| `docs/long_running_session_watcher_handoff.md` | +32 |
| **Total** | **691** |

Diff-budget override: 691 lines against a 400-line soft cap. Runtime change is 234 lines; the other two thirds are the regression tests the contract names and
the plan. Splitting tests from the behavior they pin would leave a window where
a cross-profile resume silently discards an arc with nothing to catch it, which
is the defect this slice exists to prevent.

# PR-Codex-Wake-Profile-Isolation

Contract only. This plan is committed and reviewed before any implementation
lands, per the operator's contract-first rule. No runtime file changes in this
commit.

## Why this slice exists

PR #2525 made wake turns resume one persistent Codex thread per watcher, which
fixed continuity. It did not make a wake cheap, and measurement after it merged
shows why: almost nothing a wake pays for is the conversation.

Measured on this host against codex-cli 0.155.1, using one identical prompt
(`git rev-parse --short HEAD`) and one identical follow-up that asks the agent
to recall its own previous answer from memory:

| Profile | Wake 1 | Wake 2 | Growth per wake |
|---|---:|---:|---:|
| Operator's interactive profile (today) | 75,685 | 163,194 | +87,509 |
| Dedicated `CODEX_HOME` only | 49,728 | -- | -- |
| Plus plugin and project-doc settings | 34,220 | 51,431 | +17,211 |
| Plus a wake-only skills root | 27,495 | 41,357 | +13,862 |

The whole two-turn conversation in those runs is about 90 tokens. Reading the
stored session (`~/.codex*/sessions/.../rollout-*.jsonl`) for a wake-1 turn
under the interactive profile accounts for the rest:

| Injected item | tokens |
|---|---:|
| `<recommended_plugins>`, a list of plugins that are NOT installed | 15,615 |
| `agents_md.text`, global plus project, capped at `project_doc_max_bytes` | 15,283 |
| Memory-folder guidance | 10,178 |
| Skills catalogue, as a developer message | 5,531 |
| Skills catalogue again, inside `world_state` | 5,548 |
| The prompt and the reply | 29 |

A wake uses none of it. There is no plugin to install unattended, no memory
folder to browse, and of the 26 skills in the catalogue at most three
(`pr-contract`, `reviewer-session`, `coder-session`) relate to working a PR.

### Problem-derived contract

Derived from the problem before looking at what a fix would touch.

**Root cause.** `scripts/codex_wake_run.py` launches `codex` as a child process
without deciding what profile that child runs under, so the child inherits the
operator's interactive one. Codex resolves its injected context from two
directories, and each is read from a different environment variable:

- `$CODEX_HOME` supplies `config.toml`, the memory folder, `AGENTS.md`, and a
  `skills/.system` catalogue Codex regenerates on demand.
- `$HOME/.agents/skills` supplies the shared skills catalogue.

Neither is a per-invocation flag. Verified by searching the codex binary: the
only home override string present is `CODEX_HOME`, and there is no key for a
skills root, a skills allowlist, or a per-skill disable outside of plugins. So
an unattended wake cannot opt out of the interactive surface by configuration
alone. It has to be launched under a different profile.

This is the root, not a symptom. The symptom is a large token bill; the cause
is that the runner never decides the child's profile, and the default decision
is "whatever the operator's shell had".

**What a correct fix must touch.** The single place where the runner builds the
child environment for `codex`. It must set both variables, because one root
lives under each, and setting only `CODEX_HOME` demonstrably leaves the
26-entry shared catalogue in place (the 49,728 row above).

**What must not change.** The argv shape for `exec` and `exec resume`; thread
persistence and resume; the wake lock; turn containment; the thread-id,
agent-message and usage receipts; and every exit code. The operator's real
`~/.codex` and `~/.agents` must not be written to by a wake.

## Scope (this PR)

Ownership lane: dev-workflow/codex-wake-resume
Slice phase: Workflow/process

This PR is the contract half of that phase: it adds the plan and no code.

1. Add `plans/PR-Codex-Wake-Profile-Isolation.md`, the contract below, and
   nothing else.
2. No runtime, test, doc or workflow file changes. Implementation begins only
   after the operator accepts this contract, in a separate commit on this
   branch.

### Files touched

- `plans/PR-Codex-Wake-Profile-Isolation.md` (new).

### Review Contract

The reviewer reviews the contract itself here, not an implementation. Each
criterion is about whether the contract is decidable and complete enough that
the implementation can be reviewed against it.

- The contract names both environment variables and says why one alone is
  insufficient -- settled by the 49,728 row above, measured with `CODEX_HOME`
  set and `HOME` left alone, which still carried all 26 skills.
- The contract states the default when nothing is configured, and that default
  is "behave exactly as today" -- settled by invariant I2 below, which the
  implementation must prove with a test asserting the child environment is
  unmodified when no profile is configured.
- The contract states a fail-closed rule for a misconfigured profile, and
  fail-closed means "no turn is spent" rather than "fall back to the
  interactive profile" -- settled by invariant I3 and failure case F1.
- The contract says where the effective profile is recorded, so an operator
  reading a wake log after the fact can tell which profile a turn ran under --
  settled by invariant I5.
- The contract names the credential and tooling reachability requirement, so a
  wake under an isolated `HOME` cannot silently lose git identity or `gh` auth
  -- settled by invariant I4 and failure case F3.
- The contract does not change any behavior PR #2525 settled -- settled by the
  "what must not change" list above, which the implementation must leave green
  in `tests/test_codex_wake_run.py` (365 tests at the time of writing).

**Reachability proof.** The surface is the existing entrypoint chain
`atlas-pr-webhook-receiver -> atlas-pr-watch-event -> codex_wake_bridge.py
-> CODEX_WAKE_COMMAND -> atlas-codex-wake-run`. The observable output that will
prove the wiring is the `usage=` line already written to
`~/.local/state/atlas-pr-watchers/<id>.codex-wake.log`: under an isolated
profile its `input_tokens` must land near the 27,495 row, not the 75,685 row,
for an equivalent first wake.

**Affected surfaces (at implementation time).** `scripts/codex_wake_run.py`,
`tests/test_codex_wake_run.py`, `docs/long_running_session_watcher_handoff.md`,
and possibly `scripts/install_codex_wake_bridge.py` if the profile is
installed rather than operator-provided.

**Risk areas.** Silently falling back to the interactive profile; an isolated
`HOME` that breaks git, `gh` or ssh for the woken agent; leaking the operator's
credentials into a second on-disk copy; a profile directory that a wake can
write to and corrupt for later wakes.

**Reviewer rules triggered.** R1 (plan/contract), R8 (fail-closed boundary),
R13 (receipts and audit trail).

## Mechanism

The contract, stated as observable behavior. The implementation is free to
choose its own structure as long as these hold.

**Configuration.** Two new optional runner arguments, passed as argv rather
than inherited from the ambient environment:

```text
atlas-codex-wake-run --watcher-id <id> --repo-dir <dir> \
    [--codex-home <dir>] [--agent-home <dir>]
```

Argv rather than environment is deliberate. The runner already logs its argv as
the receipt for what a wake did, and a profile chosen from ambient environment
would not appear there, so an operator reading the log could not tell which
profile an unattended turn ran under. `CODEX_WAKE_COMMAND` is argv already, so
the two compose with what exists.

**Invariants.**

- **I1.** When both arguments are given, the child `codex` process runs with
  `CODEX_HOME=<codex-home>` and `HOME=<agent-home>`, and no other environment
  variable is added, removed or altered by this feature.
- **I2.** When neither is given, the child environment is byte-identical to
  today's. This feature is opt-in and cannot change an existing deployment.
- **I3.** Either argument may be given alone, and each takes effect alone. The
  runner does not infer one from the other, because they are independent roots
  and a partial isolation is a legitimate operator choice.
- **I4.** The runner does not copy, create or write credentials. Whatever
  authentication the isolated profile has is what the operator put there.
- **I5.** The effective profile is recorded in the wake log for every turn that
  launches, alongside the existing `argv=` line, including when no profile is
  configured.
- **I6.** A wake never writes to a path outside its configured profile and its
  existing state directory. Codex's own writes inside the profile
  (`sessions/`, a regenerated `skills/.system`) are expected and permitted.

**Failure cases.**

- **F1.** A configured profile directory that does not exist, or exists and is
  not a directory, is refused **before launch** with a distinct non-zero exit
  code and no turn spent. Falling back to the interactive profile would
  reintroduce the exact defect this slice exists to remove, and doing it
  silently would hide it. Refusing before launch is safe to retry, which is the
  same rule the runner already applies to an unusable thread path.
- **F2.** A configured profile directory that exists but cannot be read is
  treated as F1.
- **F3.** A profile whose authentication is missing or invalid is **not**
  detectable before launch without spending a turn, so it is not pre-checked.
  Codex fails, the turn carries its own non-zero exit code, and the existing
  receipts record it. The contract's obligation is that this failure is loud in
  the wake log, not that it is prevented.
- **F4.** An isolated `HOME` missing git identity or `gh` credentials is an
  operator configuration error, not a runner error. The runner does not probe
  for them, because probing would either spend a turn or hard-code assumptions
  about which tools a given arc needs. The documentation obligation is in
  Deferred.

**Concurrency model.** Unchanged. The per-watcher wake lock continues to
serialize wakes, and profile directories are shared read-mostly state that
Codex itself manages. Two watchers may share one profile directory; Codex
already tolerates concurrent sessions in one `CODEX_HOME`, and nothing in this
slice adds cross-wake coordination.

**Settling test evidence the implementation must produce.**

- A test asserting the child environment carries both variables when both are
  given, one when one is given, and neither when neither is given.
- A test asserting a missing profile directory refuses before launch, with the
  fake Codex binary proving it was never invoked.
- A test asserting the wake log names the effective profile.
- The existing suite green, with no change to argv shape, resume behavior,
  receipts or exit codes.
- One live run against the real codex-cli recording `input_tokens` for an
  isolated first wake, to confirm the measured saving survives the wiring.

## Intentional

- **Two arguments, not one.** A single `--profile` that set both would be
  tidier and is wrong: they are independent roots, `CODEX_HOME` is a real Codex
  variable while `HOME` is the whole process's home, and an operator may
  reasonably isolate the Codex profile while leaving `HOME` alone.
- **No `--no-profile` escape hatch.** Absence of the arguments already is the
  escape hatch, and a third state would only create a way to disagree with
  itself.
- **The runner does not create the profile.** Creating a directory on demand
  would produce an unauthenticated profile that fails on first use after
  spending a turn, which is worse than refusing.
- **No attempt to disable Codex's built-in skills.** Measured: deleting
  `skills/.system` from an isolated `CODEX_HOME` causes Codex to regenerate it
  on the next run. The `skip_host_skill_discovery` feature flag is the right
  lever, is marked "under development", and measurably does nothing today.
  Those five built-ins are accepted as a floor.
- **Profile contents are the operator's, not the repo's.** This slice gives the
  runner the ability to use an isolated profile. What belongs inside one is a
  configuration question with a security dimension (which credentials a
  headless agent may reach), and it is recorded in Deferred rather than decided
  here.

## Deferred

Parking predicate: this slice parks profile *provisioning* and *documentation*
of a recommended profile layout, and parks anything that changes what a wake
agent is permitted to reach. It does not park correctness of the mechanism
itself.

- **Installer support for a wake profile.** `scripts/install_codex_wake_bridge.py`
  could create and drift-check a recommended profile the way it does the runner
  today. Unlocked once the contract for what the profile contains is accepted.
- **Documented profile layout.** The measured settings that produced the 27,495
  row (`project_doc_max_bytes`, the plugin features, a three-skill root, a short
  wake `AGENTS.md`) belong in
  `docs/long_running_session_watcher_handoff.md` as a recommended layout with
  the numbers. Deferred because the numbers should be re-measured against the
  wired implementation, not copied from a hand-run probe.
- **Sandbox posture under an isolated profile.** This host has
  `kernel.apparmor_restrict_unprivileged_userns=1`, so Codex's bubblewrap
  sandbox cannot create a network namespace and every sandboxed mode fails at
  its first command. The operator's interactive config works around this with
  `sandbox_mode = "danger-full-access"`. Deciding what posture an unattended
  wake gets is a security decision for the operator and is not made here.
- **A turn or token ceiling per thread.** Isolation flattens growth from
  +87,509 to +13,862 per wake but does not remove it. A ceiling that forks a
  fresh thread is still the eventual answer; the threshold wants data from a
  real multi-day arc. Carried forward from PR #2525's Deferred.

Parked hardening: none.

## Verification

This commit changes no runtime file, so there is nothing to run against it. The
measurements the contract rests on were taken by hand against the real
codex-cli 0.155.1 and are reproducible:

- Command: `CODEX_HOME=<profile> HOME=<agent-home> ~/.local/bin/atlas-codex-wake-run --watcher-id <id> --repo-dir <repo> --state-dir <dir> --sandbox danger-full-access` with the two prompts named above - Result: pass - Environment: local
- Command: `python3 -c "import json; [print(json.loads(l).get('type')) for l in open('<rollout>.jsonl')]"` to attribute injected context per record - Result: pass - Environment: local
- Command: `codex features list` to confirm which skill flags exist and their status - Result: pass - Environment: local

The implementation commit will carry its own Verification block with the test
counts and one live `usage=` line.

## Estimated diff size

| File | +/- |
|---|---:|
| `plans/PR-Codex-Wake-Profile-Isolation.md` | +288 |
| **Total** | **288** |

Contract only. The implementation that follows is budgeted at roughly 150 lines
of runtime and test change, well inside the 400-line soft cap.

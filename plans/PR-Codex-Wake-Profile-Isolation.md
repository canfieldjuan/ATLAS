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
the implementation can be reviewed against it. Every behavioral claim in this
contract was reproduced against codex-cli 0.155.1 through the installed runner
before it was written down; a reviewer can re-run each one.

- The contract names both environment variables and says why one alone is
  insufficient -- settled by the 49,728 row above, measured with `CODEX_HOME`
  set and `HOME` left alone, which still carried all 26 skills.
- The contract states the default when nothing is configured, and that default
  is "behave exactly as today" -- settled by invariant I2, which the
  implementation must prove with a test asserting the child environment is
  unmodified when no profile is configured.
- The contract requires the receipt to name the profile a turn actually ran
  under, not merely the arguments it was passed -- settled by invariant I5 and
  its reproduction, where an ambient `CODEX_HOME` and `HOME` produced output
  byte-identical to no profile at all.
- The contract assigns profile validation to Codex rather than to the runner,
  and gives the evidence for that assignment -- settled by failure cases F1 and
  F2, which show Codex refusing a nonexistent profile before any model call and
  a credential-less profile at the API, both without billing tokens and neither
  falling back to the interactive profile.
- The contract forbids a pre-launch validation check rather than leaving it
  optional -- settled by the "no pre-launch profile validation" paragraph, so a
  reviewer can reject an implementation that adds one.
- The contract states that a profile directory is written to by Codex and must
  therefore be writable -- settled by invariant I6 and its reproduction, where
  an empty profile gained `installation_id` and several sqlite databases.
- The contract names the tooling reachability requirement, so a wake under an
  isolated `HOME` cannot silently lose git identity or `gh` auth -- settled by
  failure case F3 and the Deferred documentation item.
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
- **I3.** Either argument may be given alone and each takes effect alone, and
  when only one is given the runner records that the other root is **not**
  isolated. Partial isolation measurably under-delivers: `CODEX_HOME` alone
  still carried all 26 shared skills and cost 49,728 tokens against 27,495 for
  both. Reusing an existing `CODEX_HOME` without relocating `HOME` is a real
  operator position, so it is allowed; it must not look like full isolation in
  the log.
- **I4.** The runner does not create, copy or write any part of a profile, and
  does not write credentials. A profile is an operator-supplied input.
- **I5.** For every turn that launches, the runner records the **effective**
  `CODEX_HOME` and `HOME` the child actually receives, whatever their origin --
  passed by argument, inherited from the runner's own environment, or unset.
  Recording only the arguments would leave the log blind in exactly the case
  that matters. Reproduced against the shipped runner: with
  `CODEX_HOME` and `HOME` both set in the environment and neither passed as an
  argument, the runner's output is byte-identical to a run with no profile at
  all, so nothing in the receipt distinguishes them.
- **I6.** The runner writes nothing outside its own state directory. Codex
  writes inside the profile, so a profile directory must be writable by the
  wake: reproduced by pointing `CODEX_HOME` at an empty directory, after which
  Codex created `installation_id` and several sqlite databases there.
- **I7.** The runner never selects or substitutes a profile. It passes what it
  was given, or nothing. There is no fallback path to select, which is the
  fail-closed rule enforced by construction rather than by a check.

**Failure cases.** Each of these was reproduced against codex-cli 0.155.1
through the installed runner before being written down.

- **F1. `CODEX_HOME` points at a path that does not exist.** Codex refuses
  before any model call, naming the path:
  `Error finding codex home: CODEX_HOME points to "..." but that path does not
  exist`. It does **not** create the directory and does **not** fall back to
  the interactive profile. The runner records `turn complete exit=1` and
  `usage=unavailable`. No tokens are billed.
- **F2. `CODEX_HOME` exists but carries no credential.** Codex populates the
  directory with its own scaffolding, then fails `401 Unauthorized` after
  retrying the websocket three times over about twelve seconds. Exit 1, no
  tokens billed.
- **F3. An isolated `HOME` missing git identity or `gh` credentials** is an
  operator configuration error, not a runner error. The runner does not probe
  for them, because probing would either spend a turn or hard-code assumptions
  about which tools a given arc needs. The documentation obligation is in
  Deferred.

**The runner performs no pre-launch profile validation.** This reverses the
first draft of this contract, which specified a pre-launch check that refused a
missing or unreadable profile so that no turn would be spent. Reproducing F1 and
F2 shows that check would protect against nothing: Codex already fails closed,
already refuses to fall back, and already costs nothing when it does. A runner
check would duplicate authority that belongs to Codex, could false-reject a
profile Codex would have accepted, and would introduce a time-of-check race the
runner cannot close, because a profile is handed to a child process as a string
in an environment variable and cannot be pinned to a descriptor the way PR #2525
pinned the thread file. An implementation that adds such a check should be
rejected, and an implementation that claims to close that race is overclaiming.

**Concurrency model.** The per-watcher wake lock continues to serialize wakes
and this slice adds no cross-wake coordination. One correction to the first
draft, which called a profile "read-mostly": reproducing F2 shows Codex writes
sqlite databases and an installation id inside `CODEX_HOME`, so two watchers
pointed at one profile share that state. Sharing is permitted and untested at
concurrency; a watcher that wants isolation from another watcher's Codex state
should be given its own profile directory. This is a documentation obligation,
recorded in Deferred, not a runner behavior.

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
- **No pre-launch profile validation, reversing this contract's first draft.**
  The draft specified refusing a missing or unreadable profile before launch so
  that no turn would be spent. Reproducing the two cases showed the check would
  protect against nothing: a nonexistent `CODEX_HOME` makes Codex exit before
  any model call, and a credential-less one fails at the API, both free and
  neither falling back. The check would have duplicated Codex's authority,
  risked false-rejecting a profile Codex accepts, and added a time-of-check
  race the runner cannot close. Removed on evidence.
- **A single `--wake-profile <dir>` prescribing `<dir>/codex` and `<dir>/home`
  was rejected.** It would remove the partial-isolation foot-gun by
  construction, which is its appeal, but it forces a directory layout on the
  operator and makes an existing `CODEX_HOME` unusable without moving it. The
  foot-gun is handled by I3's recording requirement instead, which costs a log
  line rather than an imposed layout.
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

This commit changes no runtime file. What it carries instead is a set of
reproductions run against the real codex-cli 0.155.1 through the installed
runner, each of which a reviewer can repeat:

- Command: `echo x | CODEX_HOME=<profile> HOME=<agent-home> ~/.local/bin/atlas-codex-wake-run --watcher-id r1 --repo-dir <repo> --state-dir <dir> --sandbox read-only --dry-run`, run once with the two variables set and once without - Result: pass - Environment: local
- Command: `echo "Reply with only the word OK." | CODEX_HOME=/tmp/codex-home-does-not-exist-xyz ~/.local/bin/atlas-codex-wake-run --watcher-id c1 --repo-dir <repo> --state-dir <dir> --sandbox read-only` - Result: fail - Environment: local
- Command: `echo "Reply with only the word OK." | CODEX_HOME=<empty dir> ~/.local/bin/atlas-codex-wake-run --watcher-id d1 --repo-dir <repo> --state-dir <dir> --sandbox read-only` - Result: fail - Environment: local
- Command: `bash scripts/check_ascii_python.sh` - Result: pass - Environment: local

The two `fail` results are the expected ones and are the evidence for F1 and F2:
each exits 1, records `usage=unavailable`, bills nothing, and does not fall back
to the interactive profile. The implementation commit will carry its own
Verification block with test counts and one live `usage=` line from an isolated
wake.

## Estimated diff size

| File | +/- |
|---|---:|
| `plans/PR-Codex-Wake-Profile-Isolation.md` | +343 |
| **Total** | **343** |

Contract only. The implementation that follows is budgeted at roughly 150 lines
of runtime and test change, well inside the 400-line soft cap.

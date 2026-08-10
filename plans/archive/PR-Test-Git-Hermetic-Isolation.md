# PR-Test-Git-Hermetic-Isolation

## Why this slice exists

`core.hooksPath` set in the developer's **global** git config applies to every
repository on the machine, including the throwaway repos the test suite builds
under `/tmp`. When a gated test pushed inside one of those repos, the managed
pre-push hook fired, ran `scripts/local_pr_review.sh`, which ran
`scripts/check_unit_gate.py` → `pytest`, whose tests push again — re-entering the
same hook. The recursion is unbounded and nests one process chain per cycle.

Observed live: **56,192 tasks** (`/proc/loadavg`), load average **38.09**, and
**59 Gi of 61 Gi** RAM consumed with swap fully exhausted. The machine was
minutes from unusable.

The hook already carried a guard, `ATLAS_SKIP_LOCAL_PR_REVIEW`, but
`run_local_unit_gate_mirror` did not set it for the unit-gate child. A normal
developer shell does not carry that guard, so the child that performed the
recursing push reached the managed hook with no skip signal.

This slice is over the 400 LOC target because the safety change crosses the
suite-wide pytest environment, the local unit-gate mirror, hook-behavior tests
that must opt back in, and failing-first regression coverage for both
file-backed and command-scope Git config. Splitting any of those surfaces would
leave one side of the recursion path unproven.

### Problem-derived contract

A test must never execute the developer's git hooks or read their git config,
regardless of machine state — and the unit gate must not hand its child an
environment that can re-enter the hook. Neither may be achieved by disabling
hooks generally: tests that assert managed-hook behavior must keep working.

## Scope (this PR)

Ownership lane: local-pr-review-tooling
Slice phase: production hardening
Max files: 6

- `tests/conftest.py`: point `GIT_CONFIG_GLOBAL` and `GIT_CONFIG_SYSTEM` at
  `os.devnull` at import time, so no user or system git config — including
  `core.hooksPath` — reaches any test. Clear command-scope Git config injection
  variables (`GIT_CONFIG_COUNT`, `GIT_CONFIG_PARAMETERS`, and numbered
  `GIT_CONFIG_KEY_*`/`GIT_CONFIG_VALUE_*` entries) for the same reason. Clear
  `GIT_TEMPLATE_DIR` because it can copy hook files into repos during
  `git init`. Supply a git identity via `GIT_AUTHOR_*`/`GIT_COMMITTER_*` to
  compensate, and preserve any ambient `ATLAS_SKIP_LOCAL_PR_REVIEW` so the unit
  gate's guard survives into pytest.
- `scripts/local_pr_review.sh`: add `ATLAS_SKIP_LOCAL_PR_REVIEW=1` to
  `unit_gate_env_values`, so the guard survives into pytest and any push it
  performs.
- `tests/test_local_pr_review.py`: eleven regression tests, covering both sides
  of the guard, guard preservation, inherited file-backed Git config, inherited
  command-scope Git config, inherited Git template hooks, and a real pytest
  launch.
- `tests/test_install_local_pr_hook.py`, `tests/test_push_pr_wrapper.py`:
  autouse fixtures clearing the guard for the modules that assert the hook runs.

### Review Contract

**Acceptance criteria** (check one by one):

1. A `core.hooksPath` configured in user/system git config does not fire inside
   a repo built by a test — proven by `test_tests_do_not_inherit_a_global_hooks_path`,
   `test_inherited_git_config_global_is_cleared_during_real_pytest_launch`, and
   `test_inherited_git_config_system_is_cleared_during_real_pytest_launch`.
2. A `core.hooksPath` injected through command-scope Git config environment
   variables does not fire inside a repo built by a test — proven by
   `test_command_scope_hooks_path_is_cleared_during_real_pytest_launch`.
3. A `pre-push` hook inherited through `GIT_TEMPLATE_DIR` is not copied into a
   repo built by a test — proven by
   `test_inherited_git_template_dir_is_cleared_during_real_pytest_launch`.
4. The same hook still fires when a caller opts back in, so the isolation is
   what suppresses it — `test_global_hooks_path_still_fires_when_not_isolated`.
5. `scripts/local_pr_review.sh` hands `ATLAS_SKIP_LOCAL_PR_REVIEW=1` to the unit
   gate, asserted against an ambient `0` so only the script can satisfy it —
   `test_local_pr_review_unit_gate_mirror_sets_recursion_guard`.
6. That guard survives a real `pytest` launch and conftest import, not just a
   stub that prints its environment —
   `test_recursion_guard_survives_a_real_pytest_launch`.
7. Tests that assert the managed hook *runs* still pass with the guard ambient,
   as the unit gate invokes them — `test_install_local_pr_hook.py` and
   `test_push_pr_wrapper.py`, both ambient states.
8. Git identity still resolves with user/system config neutralized, for the six
   committing test files that set none locally.

**Affected surfaces**

- `tests/conftest.py` — suite-wide process environment for every pytest run,
  local and CI. Highest-blast-radius file in the diff.
- `scripts/local_pr_review.sh` — the local unit-gate mirror, invoked by the
  managed pre-push hook.
- `tests/test_install_local_pr_hook.py`, `tests/test_push_pr_wrapper.py` —
  module-scoped env fixtures only; no assertion or production path changed.
- No `atlas_brain/` runtime, migration, workflow, billing, or public contract
  surface is touched, so no reachability proof is owed.

**Risk areas**

- Neutralizing global git config removes `user.name`/`user.email`; a fixture
  that commits without a local identity would break. Compensated by
  `GIT_AUTHOR_*`/`GIT_COMMITTER_*` and verified against the six files that rely
  on it.
- Preserving the guard makes the managed hook skip for the tests that assert it
  runs. Handled by clearing it in those two modules via autouse fixtures rather
  than dropping it globally, so every other test stays protected.
- `conftest.py` changes cannot be validated by running the edited files alone;
  the first revision of this PR passed its targeted runs and still regressed 10
  tests under the real gate.

**Triggered reviewer rules**

- **R2 (test evidence)** — the diff is predominantly tests, and both guards
  carry a failure-direction case, per `AGENTS.md` 3h/3i.
- **R11 (dependencies and config)** — `conftest.py` changes env/config
  resolution for the whole suite.
- **R12 (deployment safety and CI enrollment)** — `local_pr_review.sh` is the
  local mirror of the CI unit gate; both must stay in step.
- **R10 (maintainability)** — hook-behavior control moves from ambient
  environment to explicit per-module fixtures.
- **R14 (verify against the codebase)** — universal.

### Files touched

- `plans/PR-Test-Git-Hermetic-Isolation.md`
- `scripts/local_pr_review.sh`
- `tests/conftest.py`
- `tests/test_install_local_pr_hook.py`
- `tests/test_local_pr_review.py`
- `tests/test_push_pr_wrapper.py`

### Boundary-change enumeration
No module or ownership boundary changes. No new imports, callers, or public
surfaces. `conftest.py` gains only process-environment assignments in the
existing import-time block; `local_pr_review.sh` gains one array entry.

### Git config closure declaration

The file-backed Git config override set is **closed** for this slice:
`GIT_CONFIG_GLOBAL` and `GIT_CONFIG_SYSTEM` are the complete file-backed
environment controls and are assigned to `os.devnull`. The canonical source is
`tests/conftest.py`.

The command-scope Git config injection set is also **closed** for this slice.
The canonical exact-name source is
`tests/conftest.py::_GIT_CONFIG_INJECTION_ENV_NAMES`
(`GIT_CONFIG_COUNT`, `GIT_CONFIG_PARAMETERS`). The canonical prefix source is
`tests/conftest.py::_GIT_CONFIG_INJECTION_ENV_PREFIXES`
(`GIT_CONFIG_KEY_`, `GIT_CONFIG_VALUE_`) for Git's numbered key/value pairs.

Unlisted `GIT_CONFIG_*` variables are not treated as Git config-injection
controls by this PR and remain unchanged. If Git adds or documents another
environment variable that injects config into commands, that is a future slice
only after a failing proof or a newly observed recurrence. The implementation is
bound to those constants, and the regression tests exercise the file-backed
source (`GIT_CONFIG_GLOBAL`, `GIT_CONFIG_SYSTEM`) plus the command-scope
exact/prefix source (`GIT_CONFIG_COUNT`, `GIT_CONFIG_KEY_0`,
`GIT_CONFIG_VALUE_0`, `GIT_CONFIG_PARAMETERS`) through real nested pytest
launches.

The hook-producing Git template environment set is **closed** for this slice:
`GIT_TEMPLATE_DIR` is the complete inherited template-directory control this PR
handles. It is not a Git config variable, so it is documented separately from
the `GIT_CONFIG_*` closure above. Unlisted template/config-path mechanisms are
future work only after a failing proof or newly observed recurrence.

### Deployed-config probing
No deployed-config, env, secret, or blueprint changes. `GIT_CONFIG_GLOBAL` and
`GIT_CONFIG_SYSTEM` are set only inside the pytest process and its children.

## Mechanism

Git resolves `core.hooksPath` from the full config stack, so a value in the
user's global config applies to a repo created seconds earlier by `git init` in
`/tmp`. Pointing `GIT_CONFIG_GLOBAL`/`GIT_CONFIG_SYSTEM` at `/dev/null` makes git
read no user or system config at all, which removes the vector structurally
rather than blacklisting one key.

That also removes `user.name`/`user.email`. Most git fixtures set an identity
locally, but six committing test files do not
(`test_check_ai_reconciliation_live`, `test_check_seam_convergence`,
`test_getapp_parser`, `test_pr_watcher`, `test_security_guardrails_workflow`,
`test_watch_owned_pr`), so identity is supplied via environment variables, which
apply with no config file present.

For the second layer, `unit_gate_env_values` is passed to all three
`check_unit_gate.py` invocations, and `check_unit_gate.py:336` calls
`subprocess.run` with no `env=`, so the value inherits cleanly through to pytest
and any git it spawns.

Git can also receive command-scope config from inherited environment variables:
the closed exact-name set (`GIT_CONFIG_COUNT`, `GIT_CONFIG_PARAMETERS`) and the
closed numbered key/value prefix set (`GIT_CONFIG_KEY_*`,
`GIT_CONFIG_VALUE_*`). Conftest clears those names through the canonical
constants before tests run, so a `core.hooksPath` injected that way cannot reach
a throwaway repo created by pytest.

Git templates are a third hook-producing path: `GIT_TEMPLATE_DIR` can cause
plain `git init` to copy an executable `hooks/pre-push` into a fixture repo
before the first push. Conftest clears that variable at import time, and the
nested-pytest regression launches with a hook-bearing template directory to
prove the inner fixture repo does not copy or run it.

## Intentional

- `conftest.py` **preserves** `ATLAS_SKIP_LOCAL_PR_REVIEW` rather than dropping
  it, so the guard the unit gate supplies survives into pytest. Both
  `scripts/push_pr.sh:80` and the installed hook branch on that variable, so ten
  tests that assert the hook *runs* would see it skip; those two modules clear
  it with an autouse fixture instead. Dropping it suite-wide — the first
  revision of this PR — traded protection for every test to accommodate ten, and
  left the recursion's own crossing point (checker → pytest) unguarded.
- The guard is asserted through a **real pytest launch**
  (`test_recursion_guard_survives_a_real_pytest_launch`), not only against a
  stub checker that prints its environment. The stub never imports conftest, so
  it cannot observe whether conftest preserves or deletes the value — the exact
  step where the first revision went wrong.
- Identity vars use `setdefault`, so a fixture can still override them; the
  `GIT_CONFIG_*` vars are assigned unconditionally because hermetic git is a
  guarantee, not a default a stale environment may override.

## Deferred

- Future PR: machine-local temp-repo hook hard-stop. Unlock condition: the
  repo-managed pytest isolation in this PR fails to prevent a recurrence, or
  the operator explicitly wants belt-and-suspenders protection outside repo
  content. Parking predicate: `~/.claude/hooks/git/pre-push` is machine-local
  configuration, not repo content, and the current repo tests now prove
  inherited file-backed and command-scope hook config is neutralized before a
  test push.
- Future refactor: consolidate duplicated git fixture helpers
  (`_write_fixture_repo`, `_git`). Unlock condition: #2341 has landed and an
  adjacent PR needs shared fixture-helper edits. Parking predicate: helper
  consolidation is maintenance cleanup, not required to stop the recursion or
  prove the hermetic Git-config boundary.
- Parked hardening: none.

## Verification

- The two new failing-first tests were confirmed red against `origin/main`:
  the isolation test failed with `global pre-push hook ran inside a test repo`,
  reproducing the incident inside a test. The control passed on `main`, proving
  the mechanism is real before any fix.
- After the change: all file-backed and command-scope hook isolation tests pass.
- The first attempt at the guard regressed **10 tests** in the full unit gate
  (8 in `test_install_local_pr_hook.py`, 2 in `test_push_pr_wrapper.py`), all
  asserting the managed hook runs. Caught by running the real gate, not just
  the edited files. Round 1 keeps the guard and clears it per module instead;
  the trio passes in **both** ambient states — **68 passed** with
  `ATLAS_SKIP_LOCAL_PR_REVIEW=1` set, as the unit gate invokes them, and
  **68 passed** without it.
- Blast-site files (`test_local_pr_review.py`, `test_open_pr_wrapper.py`,
  `test_push_pr_wrapper.py`): **109 passed**.
- Identity regression, the six files that commit without a local identity:
  `pytest tests/test_check_ai_reconciliation_live.py tests/test_check_seam_convergence.py tests/test_getapp_parser.py tests/test_pr_watcher.py tests/test_security_guardrails_workflow.py tests/test_watch_owned_pr.py`
  -- **282 passed**.
- Full git-touching sweep, 33 test modules plus the `tests/conftest.py`
  blast-site imported by each run:
  `pytest tests/test_audit_cross_layer_callers.py tests/test_audit_fix_loop_disposition.py tests/test_audit_plan_code_consistency.py tests/test_audit_plan_doc_diff_size.py tests/test_audit_plan_doc_files_touched.py tests/test_audit_pr_body.py tests/test_audit_pr_plan_presence.py tests/test_audit_pr_session_drift.py tests/test_check_ai_reconciliation_live.py tests/test_check_boundary_change_enumeration.py tests/test_check_deployed_config_probing.py tests/test_check_guard_class_closure.py tests/test_codex_issue_queue.py tests/test_content_factory_copy_verification.py tests/test_content_factory_runner.py tests/test_content_factory_store.py tests/test_detect_retired_failure_modes.py tests/test_docs_no_raw_deflection_request_ids.py tests/test_install_local_pr_hook.py tests/test_leads_intake.py tests/test_local_pr_review.py tests/test_new_pr_plan.py tests/test_open_pr_wrapper.py tests/test_plan_admission_workflow.py tests/test_pr_watcher.py tests/test_pre_push_audit.py tests/test_pre_push_audit_workflow.py tests/test_push_pr_wrapper.py tests/test_security_policy_docs.py tests/test_session_lane_workflow.py tests/test_sync_pr_plan.py tests/test_unit_gate_selector_fallback.py tests/test_update_pr_body_wrapper.py`
  -- **1,828 passed**.
- Current focused review-fix verification:
  `pytest tests/test_local_pr_review.py -k "recursion_guard or global_hooks_path or command_scope_hooks_path or inherited_git_config or git_template_dir" -q`
  -- **11 passed, 31 deselected**;
  `ATLAS_SKIP_LOCAL_PR_REVIEW=1 pytest tests/test_install_local_pr_hook.py tests/test_push_pr_wrapper.py -q`
  -- **32 passed**; and
  `pytest tests/test_local_pr_review.py tests/test_install_local_pr_hook.py tests/test_push_pr_wrapper.py -q`
  -- **74 passed**.
- Process-count sampled across the runs stayed flat (2,445 → 2,468), i.e. the
  recursion cannot re-enter.

## Estimated diff size

| File | LOC |
|---|---:|
| `plans/PR-Test-Git-Hermetic-Isolation.md` | 315 |
| `scripts/local_pr_review.sh` | 3 |
| `tests/conftest.py` | 42 |
| `tests/test_install_local_pr_hook.py` | 14 |
| `tests/test_local_pr_review.py` | 283 |
| `tests/test_push_pr_wrapper.py` | 13 |
| **Total** | **670** |

## Cold diff reconstruction

- `tests/conftest.py`: after the existing `ATLAS_DB_*` `setdefault` block, assign
  `GIT_CONFIG_GLOBAL`/`GIT_CONFIG_SYSTEM` to `os.devnull`, declare the closed
  exact-name and prefix constants for command-scope Git config injection, clear
  those variables, clear `GIT_TEMPLATE_DIR`, `setdefault` the four
  `GIT_AUTHOR_*`/`GIT_COMMITTER_*` vars, and
  `os.environ.setdefault("ATLAS_SKIP_LOCAL_PR_REVIEW", "1")` with a comment
  naming the two modules that clear it locally.
- `scripts/local_pr_review.sh`: add `ATLAS_SKIP_LOCAL_PR_REVIEW=1` as the first
  entry of `unit_gate_env_values`, with a comment naming the recursion.
- `tests/test_local_pr_review.py`: add
  `test_local_pr_review_unit_gate_mirror_sets_recursion_guard` (checker prints
  the var, ambient set to `0`),
  `test_tests_do_not_inherit_a_global_hooks_path` (fake `HOME` with a
  hooksPath-setting gitconfig; marker must not appear; push succeeds), and
  `test_global_hooks_path_still_fires_when_not_isolated` (same setup, explicit
  `GIT_CONFIG_GLOBAL`; marker must appear; push fails), and
  `test_conftest_preserves_the_recursion_guard` plus
  `test_recursion_guard_survives_a_real_pytest_launch`, which spawns a real
  pytest so conftest is genuinely imported.
- `test_command_scope_hooks_path_is_cleared_during_real_pytest_launch`, which
  launches pytest with both `GIT_CONFIG_COUNT` numbered `core.hooksPath` entries
  and `GIT_CONFIG_PARAMETERS` injected, then proves a push inside the inner test
  does not run the hook.
- `test_inherited_git_config_global_is_cleared_during_real_pytest_launch` and
  `test_inherited_git_config_system_is_cleared_during_real_pytest_launch`, which
  launch nested pytest processes with each inherited file-backed config variable
  pointing at a hook-bearing file and prove conftest overwrites them before the
  inner push.
- `test_inherited_git_template_dir_is_cleared_during_real_pytest_launch`, which
  launches nested pytest with a hook-bearing template directory and proves
  conftest clears it before the inner fixture repo is initialized and pushed.

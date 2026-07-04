# PR-Session-Bootstrap-Temporal-Discipline

## Why this slice exists

Issue #1981 tracks the Atlas-only CI temporal discipline audit follow-up. The
audit found one P2 documentation contradiction: the bootstrap's recurring-lapse
list says to check `gh pr checks` before claiming done, while the same document
and `AGENTS.md` say the builder must stop after opening or updating a PR and
wait for watcher/operator signal.

The operator also clarified the multi-agent operating reality: there are often
two or three active builder sessions. A single shared SESSION_STATE.local.md
file makes those sessions read and rewrite the same local baton in competing
ways.

Root cause: one checklist item mixed two time boundaries. It correctly reminded
builders that local passes are not CI green, but it also read like a post-push
polling instruction. The same docs also described session state as one shared
filename, not one state file per active session. This change fixes both roots:
the bootstrap wording scopes the CI reminder to pre-push local verification, and
the session-state contract now labels one ignored state file per session.

## Scope (this PR)

Ownership lane: workflow/autonomous-ci-cd-map
Slice phase: Workflow/process

1. Reword the bootstrap CI reminder so it cannot be read as "poll CI after
   push."
2. Clarify that every active builder owns its own labeled state file:
   SESSION_STATE.<session-id>.local.md, with SESSION_STATE.local.md
   retained only as a legacy single-session fallback.
3. Teach the ownership guard to default to ATLAS_SESSION_STATE_FILE so the
   docs and tooling agree.
4. Preserve the existing Atlas handoff model: a session-scoped state file plus
   watcher status is the baton; this slice does not add another marker or
   watcher script.

### Review Contract

- Acceptance criteria:
  - [ ] `docs/SESSION_BOOTSTRAP.md` no longer tells a builder session to check
        PR CI after opening or updating a PR.
  - [ ] The doc still tells builders to run scoped/local checks before push and
        not claim unrun checks passed.
  - [ ] The doc explicitly aligns post-push work with the existing
        watcher/operator handoff from `AGENTS.md` and the later bootstrap
        stop-after-push rule.
  - [ ] The optional one-shot state marker from #1981 is explicitly not added
        because Atlas already has session-scoped state files and watcher
        status.
  - [ ] Concurrent sessions are told to use distinct local files and export
        ATLAS_SESSION_STATE_FILE.
  - [ ] `scripts/check_session_pr_ownership.py` reads
        ATLAS_SESSION_STATE_FILE by default, while preserving the legacy
        fallback.
  - [ ] Wake-bridge prompts point the resumed agent to the configured state
        file rather than the shared legacy filename.
- Reachability proof: focused unit tests exercise the env-selected ownership
  file and the generated wake prompt.
- Affected surfaces: docs/process, local ownership guard, Codex wake prompt.
- Risk areas: workflow ambiguity, token burn from polling, scope drift into new
  watcher infrastructure, concurrent-session state collision.
- Reviewer rules triggered: R1, R2, R10, R14.

### Files touched

- `.gitignore`
- `AGENTS.md`
- `CLAUDE.md`
- `docs/SESSION_BOOTSTRAP.md`
- `docs/SESSION_STATE_TEMPLATE.md`
- `docs/ai_dev_operating_model.md`
- `docs/autonomous_coding_repo_playbook.md`
- `docs/ci_cd_autonomous_coding_map.md`
- `docs/ci_cd_runtime_duplication_audit.md`
- `docs/long_running_agent_monitoring_spec.md`
- `docs/long_running_session_watcher_handoff.md`
- `plans/PR-Session-Bootstrap-Temporal-Discipline.md`
- `scripts/check_session_pr_ownership.py`
- `scripts/codex_wake_bridge.py`
- `tests/test_check_session_pr_ownership.py`
- `tests/test_codex_wake_bridge.py`

## Mechanism

Replace the ambiguous "check `gh pr checks` is green before claiming done"
sentence with a two-boundary statement:

1. Before push, run the relevant local/scoped checks honestly.
2. After push/open/update, merge readiness belongs to the watcher/operator
   signal, not an in-session polling loop.

Then update the session-state convention:

1. Docs name the preferred file shape as
   SESSION_STATE.<session-id>.local.md.
2. `.gitignore` ignores both the legacy and session-labeled filenames.
3. `scripts/check_session_pr_ownership.py` resolves ATLAS_SESSION_STATE_FILE
   first, then falls back to SESSION_STATE.local.md for legacy single-session
   worktrees.
4. `scripts/codex_wake_bridge.py` includes the configured state path in its
   generated prompt and ownership-guard command.

Atlas still uses the same handoff baton -- session state plus watcher status --
so a second marker would create duplicate state instead of a cleaner workflow.

## Intentional

- No CI workflow changes; #1981 identified a wording contradiction, not a broken
  gate.
- No new watcher, timer, marker, or state-printer script; Atlas already has the
  richer handoff machinery documented in the session state file and watcher
  status.
- No product-code changes.

## Deferred

- The cross-repo one-shot state helper idea is not adopted for Atlas in this
  slice. If another repo lacks Atlas's handoff machinery, it can adopt its own
  non-polling helper there.

Parked hardening: none.

## Verification

- Ran `scripts/sync_pr_plan.py` against `plans/PR-Session-Bootstrap-Temporal-Discipline.md` -- passed.
- Ran `scripts/sync_pr_plan.py --check` against `plans/PR-Session-Bootstrap-Temporal-Discipline.md` -- passed.
- Ran pytest for `tests/test_check_session_pr_ownership.py` and `tests/test_codex_wake_bridge.py` -- 23 passed.
- Ran `scripts/check_session_pr_ownership.py` with ATLAS_SESSION_STATE_FILE set to SESSION_STATE.local.md against #1982 -- passed.
- `git diff --check --cached` -- passed.
- `scripts/push_pr.sh` runs the body-aware local review before push.

## Estimated diff size

| File | LOC |
|---|---:|
| `.gitignore` | 1 |
| `AGENTS.md` | 34 |
| `CLAUDE.md` | 4 |
| `docs/SESSION_BOOTSTRAP.md` | 9 |
| `docs/SESSION_STATE_TEMPLATE.md` | 15 |
| `docs/ai_dev_operating_model.md` | 23 |
| `docs/autonomous_coding_repo_playbook.md` | 4 |
| `docs/ci_cd_autonomous_coding_map.md` | 2 |
| `docs/ci_cd_runtime_duplication_audit.md` | 2 |
| `docs/long_running_agent_monitoring_spec.md` | 4 |
| `docs/long_running_session_watcher_handoff.md` | 38 |
| `plans/PR-Session-Bootstrap-Temporal-Discipline.md` | 154 |
| `scripts/check_session_pr_ownership.py` | 21 |
| `scripts/codex_wake_bridge.py` | 12 |
| `tests/test_check_session_pr_ownership.py` | 18 |
| `tests/test_codex_wake_bridge.py` | 4 |
| **Total** | **345** |

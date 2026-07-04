# PR-Long-Running-Watcher-Handoff

## Why this slice exists

Issue #1962 now has the S1 CI/CD map on main, and the operator asked for the
next practical handoff: a prompt they can feed other long-running builder
sessions so those sessions know the new watcher rules and local files. The root
cause is that the watcher setup currently exists as local machine infrastructure
and chat context; without a repo-visible handoff, parallel sessions can miss the
no-auto-merge rule, reuse the wrong watcher, or keep relying on the operator to
babysit green CI.

This change fixes the root for the handoff layer by documenting the per-session
watcher contract, setup commands, state meanings, and copy/paste prompt in the
repo. It does not repo-ize the watcher executable itself.

## Scope (this PR)

Ownership lane: workflow/autonomous-ci-cd-map
Slice phase: Workflow/process

1. Add a long-running session watcher handoff doc with:
   - local watcher file inventory;
   - one-watcher-per-session setup commands;
   - no-auto-merge safety rules;
   - watcher state meanings;
   - a paste-ready prompt for other builder sessions.
2. Archive the merged S1 plan from #1963 and refresh the plan index.

### Review Contract

Acceptance criteria:
- The handoff prompt tells other sessions to read the new CI/CD map and watcher
  handoff before acting.
- The handoff explicitly preserves no-auto-merge; green means
  `ready_for_human_merge`, not merge.
- The setup commands are session-scoped and require a unique `<session-id>` plus
  the owned PR head SHA.
- The doc tells sessions to update `SESSION_STATE.local.md` and respect PR
  ownership before inspecting or mutating PRs.
- This PR changes documentation/plan archive only; no workflow behavior or
  watcher executable changes.

Affected surfaces:
- Developer workflow docs for long-running Atlas builder sessions.

Risk areas:
- Workflow/process drift if the prompt implies a session may touch unowned PRs.
- Merge-safety drift if the watcher appears to authorize auto-merge.
- Stale local-file assumptions because the watcher executable is still local
  machine infrastructure.

Triggered reviewer rules:
- R1 Requirements match
- R2 Test evidence
- R6 Workflow/process
- R14 Codebase verification

### Files touched

- `docs/long_running_session_watcher_handoff.md`
- `plans/INDEX.md`
- `plans/PR-Long-Running-Watcher-Handoff.md`
- `plans/archive/PR-CI-CD-Autonomous-Map.md`

## Mechanism

`docs/long_running_session_watcher_handoff.md` turns the local watcher setup
from chat context into a repo-visible runbook. It names the home-directory files,
defines the per-session `.env` shape, gives the systemd commands, explains the
watcher states, and includes a paste-ready prompt for other builders.

The prompt references `AGENTS.md`, `CLAUDE.md`, `docs/SESSION_BOOTSTRAP.md`,
`docs/ci_cd_autonomous_coding_map.md`, this handoff doc, and
`SESSION_STATE.local.md` so a restarted or parallel session rehydrates from
durable state instead of memory.

The merged S1 plan is moved to `plans/archive/`, and `plans/INDEX.md` is
refreshed with `python scripts/archive_plans.py index`.

## Intentional

- No workflow, branch-protection, or watcher executable changes in this slice.
  The operator needs the handoff prompt now; repo-izing the helper script can be
  a follow-up portability slice.
- The prompt says no auto-merge even when the watcher reports
  `ready_for_human_merge`, because late review comments can arrive after green
  CI.
- The doc names local home-directory files because the first watcher is already
  installed locally and available to sibling sessions on this machine.

## Deferred

- Repo-owned watcher installer/executable if we want this portable across
  machines instead of only available on Juan's current workstation.
- #1962 S3: measure runtime and duplicate checks before proposing speedups.
- #1962 S4-S7: monitoring spec, reusable playbook, and Reddit/public story
  draft.

Parked hardening: none.

## Verification

- `python scripts/sync_pr_plan.py plans/PR-Long-Running-Watcher-Handoff.md --check`
  - passed.
- `bash scripts/local_pr_review.sh --current-pr-body-file /tmp/long_running_watcher_handoff_pr_body.md`
  - passed; non-blocking warning only: #1953 also edits `plans/INDEX.md`.

## Estimated diff size

| File | LOC |
|---|---:|
| `docs/long_running_session_watcher_handoff.md` | 180 |
| `plans/INDEX.md` | 3 |
| `plans/PR-Long-Running-Watcher-Handoff.md` | 117 |
| `plans/archive/PR-CI-CD-Autonomous-Map.md` | 0 |
| **Total** | **300** |

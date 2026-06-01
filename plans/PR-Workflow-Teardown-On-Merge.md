# PR-Workflow-Teardown-On-Merge

## Why this slice exists

A worktree-cleanup session (2026-06-01) found two stale-state messes that each
cost real time to untangle: a worktree pinned to local `main` that had drifted
458 commits behind `origin/main` with 172 staged files (all already landed via
PR #694), and a primary checkout carrying a ~300-file dirty index plus three
local commits that merely mirrored already-merged PRs. The root cause was not a
bug — it was the absence of a teardown discipline: branches and worktrees were
created for work that then landed via PR, but the local copies were never torn
down, so they drifted and accumulated. The companion `.gitignore` hardening
(#1219) fixes the `git clean` landmine; this slice makes the prevention rule
explicit so the mess does not recur.

## Scope (this PR)

Ownership lane: workflow/process
Slice phase: Workflow/process

1. Add AGENTS.md §1g "Teardown on merge": delete branch + worktree when a PR
   merges; `origin/main` is the only source of truth; check stale branches
   against `origin/main` before resurrecting; never `git clean -f` without a
   dry-run.
2. Add a matching item 7 to the SESSION_BOOTSTRAP.md fresh-session checklist so
   each builder session inherits the rule.

### Files touched

- `plans/PR-Workflow-Teardown-On-Merge.md`
- `AGENTS.md`
- `docs/SESSION_BOOTSTRAP.md`

## Mechanism

Docs-only. AGENTS.md gains §1g at the end of the "1. PR shape" lifecycle
(after §1f "Open ready for review"), closing the loop open -> review -> merge ->
teardown. SESSION_BOOTSTRAP.md §1 gains item 7 pointing at §1g, phrased for a
fresh builder session. Both restate the same three habits: tear down on merge,
treat local branches/worktrees as disposable mirrors of `origin/main`, and
dry-run `git clean` because untracked secret files (`.env.bak-*`,
`*.production.env`, now gitignored via #1219) live in the tree.

## Intentional

- Docs/process only; no code, no behavior change.
- `-D` (not `-d`) is named for branch deletion because squash-merge leaves the
  local branch unmerged by content and `-d` refuses it — a predictable footgun.
- The clean-safety note references the same patterns #1219 added, keeping the
  two hygiene slices consistent.

## Deferred

- A scripted cleanup helper / git aliases (dry-run-first `clean`, prune-merged
  branches) — separate tooling slice if desired.

## Parked hardening

None.

## Verification

- `git diff` is limited to the two doc files + this plan.
- AGENTS.md §1g renders under "1. PR shape" after §1f; SESSION_BOOTSTRAP.md
  item 7 renders in the §1 fresh-session checklist.
- No Python sources are touched, so the ASCII and import gates do not apply;
  the local PR review hook covers plan/code consistency and PR drift.

## Estimated diff size

| Area | LOC |
|---|---:|
| AGENTS.md §1g | ~24 |
| SESSION_BOOTSTRAP.md item 7 | ~2 |
| Plan doc | ~75 |
| **Total** | ~100 |

Docs/process only; well under the 400-LOC soft cap.

# SESSION_STATE.local.md Template

Create `SESSION_STATE.local.md` at the repository root from this template for
each builder session. Keep it local; it is ignored by git.

Update it before opening a PR, after pushing a PR update, after merging a PR,
and after any compaction/restart reorientation. If current GitHub state
conflicts with this file, stop and ask the operator instead of guessing.

```md
# Atlas Builder Session State

Last updated: YYYY-MM-DD HH:MM TZ
Session role: builder
Operator-assigned lane: <one sentence>
Current task: <one sentence>
Spark/subagent routing: used <what/why> | considered <why main/direct was better> | not applicable

## Owned Active PR

Status: none | planned | open | merged
PR: #<number or none>
Title: <title or none>
URL: <url or none>
Branch: <branch or none>
Plan: plans/PR-<Slice>.md
Expected head SHA: <sha or none>
Ownership lane: <lane from plan>
Allowed actions: inspect | update | merge-on-operator-signal | none

## PRs This Session May Touch

- #<number> <title> -- reason this session owns it

## PRs This Session Must Not Touch

- #<number> <title> -- owner/session/lane if known

## Recent PRs Merged By This Session

- #<number> <title> -- merged at <sha/time>

## Current Worktree

Path: <absolute worktree path>
Branch: <branch>
Base: origin/main @ <sha>
Dirty state expected: yes | no

## Last Safe Action

<One sentence: e.g. "Opened #1234 and stopped; waiting for operator signal.">

## PR Fix Mode (active fix loop)

Active: yes | no   (fill in only while iterating on red CI / review comments)
PR: #<number or none>
Branch: <branch>
Latest commit: <sha at last push>
Allowed files (max N=<n>):
- <path or glob -- the failure source, not everything the symptom touches>
Current failing check / comment: <e.g. live-reconciliation: copilot :194,:226>
Last useful log finding: <the single line that localized it>
Next exact action: <one sentence>
Do-NOT-redo: <paths ruled out, checks already green, dead ends>

## Resume Checklist

- [ ] Read this file before any PR action.
- [ ] If PR Fix Mode is active, read its block before any edit; do not touch
      files outside Allowed files (widening requires the §3k upstream reason).
- [ ] Run `gh pr list --state open`.
- [ ] Run `git log --oneline -15 origin/main`.
- [ ] Confirm the current PR is listed under "Owned Active PR" or "PRs This
      Session May Touch" before inspecting comments, pushing updates, or
      merging.
- [ ] Treat every other open PR as "must not touch" unless the operator
      explicitly reassigns it.
```

## Ownership Rule

If a PR is not listed as owned in `SESSION_STATE.local.md`, it is not yours.
Lane proximity is not ownership. Similar file paths are not ownership. A PR
opened by another active session is not yours unless the operator explicitly
reassigns it and the map is updated first.

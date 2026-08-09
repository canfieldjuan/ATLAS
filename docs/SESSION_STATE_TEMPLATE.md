# Session-State File Template

Create one local state file per builder session at the repository root from
this template. Prefer `SESSION_STATE.<session-id>.local.md` (for example,
`SESSION_STATE.codex-workflow-1982.local.md`). Use legacy
`SESSION_STATE.local.md` only when one active session owns the worktree.

Keep the file local; `SESSION_STATE.local.md` and
`SESSION_STATE.*.local.md` are ignored by git. Export
`ATLAS_SESSION_STATE_FILE=<absolute or repo-relative path>` for the active
session so ownership guards and wake prompts read the right file.

Update it before opening a PR, after pushing a PR update, after merging a PR,
and after any compaction/restart reorientation. If current GitHub state
conflicts with this file, stop and ask the operator instead of guessing.

```md
# Atlas Builder Session State

Last updated: YYYY-MM-DD HH:MM TZ
Session role: builder
Current lane: <one sentence>
Current task: <one sentence>
Spark/subagent routing: used <what/why> | considered <why main/direct was better> | not applicable
Builder surface: Claude Code native | Codex/local CLI | other <name>
Wake mode: Claude native PR subscription + 30-minute poll | Codex wake bridge | local watcher state-only | none

## Owned Active PR

Status: none | planned | open | merged
PR: #<number or none>
Title: <title or none>
URL: <url or none>
Branch: <branch or none>
Plan: plans/PR-<Slice>.md
Expected head SHA: <sha or none>
Ownership lane: <lane from plan>
Allowed actions: inspect | update | report-ready | active-builder-merge-after-guarded-signal | none
Standing merge authorization: <none | authorized active-builder merge by <operator/source> for <arc>; scheduled-ready-only; watcher merge forbidden>
Push/review-event hook: <name and trigger | unavailable | none>
Timer/poll hook: Claude native 30-minute poll | systemd/cron/webhook name | none
Wake bridge: Claude native subscription | <external Codex launch/resume bridge> | unavailable | none
Ready-state handoff: scripts/report_pr_watcher_state.py | <other read-only reporter> | none
Issue queue: #<issue number or none>
Operator email: <email or none>
Deferred decisions issue: #<issue number or none>
Next timer wake: <timestamp or none>
Last watcher state: <state/details or none>

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
Activation head: <sha when fix mode was armed>
Activation dirty paths:
- <repo-relative staged/working/untracked path present when armed, or none>
Allowed files (max N=<n>):
- <path or glob -- the failure source, not everything the symptom touches>
Symptom: <failing check or review claim being addressed>
Root cause: <upstream defect, not the visible leaf symptom>
Source trace: <symptom -> intermediate cause -> upstream source>
Fix strategy: upstream-root | symptom-only-deferred
Upstream files:
- <repo-relative file where the upstream source is fixed>
Symptom-only reason: <required only for symptom-only-deferred>
Follow-up: <required only for symptom-only-deferred>
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
- [ ] Confirm `Builder surface` and `Wake mode` match the actual session. Claude
      Code native sessions use Claude's PR subscription and 30-minute poll;
      Codex/local sessions need an external wake bridge for true autonomous
      resume.
- [ ] Confirm `Push/review-event hook`, `Timer/poll hook`, `Wake bridge`, `Next
      timer wake`, and `Last watcher state` reflect the current long-running
      setup before relying on autonomous wake-ups.
- [ ] Run the ready-state handoff reporter before starting a new slice in a
      Codex/local watcher arc.
- [ ] Confirm `Standing merge authorization` is explicit before the active
      builder merges on a scheduled `ready_for_human_merge` wake; do not infer
      it from watcher state.
- [ ] At technical forks, choose the durable fix and document it; defer only
      genuinely operator-owned decisions to the configured issue/email path.
- [ ] Treat every other open PR as "must not touch" unless the operator
      explicitly reassigns it.
```

## Ownership Rule

If a PR is not listed as owned in this session's state file, it is not yours.
Lane proximity is not ownership. Similar file paths are not ownership. A PR
opened by another active session is not yours unless the operator explicitly
reassigns it and the map is updated first.

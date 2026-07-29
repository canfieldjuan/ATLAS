# PR-Overnight-Arc-Workflow

## Why this slice exists

The overnight-arc workflow (unattended long-running coding task: pre-flight
contract -> night loop -> morning report) existed only as machine-local
session prose, so it could not survive compaction or reach other sessions.
Compaction survival requires the workflow to live where post-compaction
context is guaranteed: the repo contract docs. This codifies it in-repo, the
same treatment `docs/PR_RECONSTRUCTION_PROTOCOL.md` got for the review
protocol, and adds the one missing piece of machinery (a portable status-only
owned-PR watcher) that every builder session has been rewriting inline.

Diff-budget note: ~423 LOC, marginally over the 400 soft cap after review
round 1 added the watcher-safety enrollment and fail-closed watcher fixes;
the runbook + its watcher + the contract wiring must land together or the
workflow is codified in a state its own verification cannot prove.

### Problem-derived contract

- Root cause: post-compaction context is rebuilt only from the compaction
  summary plus the always-reinjected repo contract docs (`CLAUDE.md`,
  `AGENTS.md`). The overnight-arc workflow lived outside both (machine-local
  session prose), so a compacted or fresh session had no durable path back to
  the workflow or its in-flight arc state.
- Correct fix must touch/change: a canonical in-repo runbook under `docs/`;
  an `AGENTS.md` section binding assigned overnight arcs to it; a `CLAUDE.md`
  compact-instructions baton entry so mid-arc state survives compaction and
  points back to the runbook; and the watcher machinery the runbook depends
  on (`scripts/watch_owned_pr.sh`), which must be status-only under the
  existing watcher-safety rule.
- Must not change: merge-gate semantics (the AGENTS 3c.1.8 two-gate rule is
  referenced, not altered), required checks, the watcher-safety audit's
  detection semantics (its scan list is EXTENDED -- review round 1 proved the
  new watcher and runbook were not covered, so the audit passed vacuously;
  enrollment is required for the plan's own verification claim to be true),
  existing AGENTS/CLAUDE content (edits additive only), any code paths,
  tests, or product surfaces.

## Scope (this PR)

Ownership lane: workflow/process
Slice phase: Workflow/process

1. Add `docs/OVERNIGHT_ARC_WORKFLOW.md` -- the canonical runbook: mandatory
   interactive pre-flight (readiness contract + all questions asked while the
   operator is present), night-loop deltas over the AGENTS section 3 builder
   contract, true-blocker channel, mandatory morning report, kickoff prompt.
2. Add `scripts/watch_owned_pr.sh` -- portable single-PR watcher; status-only
   (reports MERGED/CLOSED, HEAD-MOVED, ACTIONABLE, MERGE-READY and exits).
   MERGE-READY is presence-based and fail-closed: every required
   branch-protection context (read at runtime from
   `scripts/check_required_status_checks.py`) present and success, plus the
   `claude-review` commit status, 0 unresolved threads (fail closed on
   unfetched thread pages), and no CHANGES_REQUESTED review decision
   (AGENTS section 3c.1.8). Definite negatives exit on any cycle.
3. Wire `AGENTS.md`: new section 3c.2 naming the runbook as governing for
   assigned overnight arcs.
4. Wire `CLAUDE.md` Compact Instructions: add the overnight baton to the
   preserve-verbatim list so a compacted overnight session resumes from the
   baton and re-reads the runbook.
5. Enroll the new watcher and runbook in `scripts/audit_pr_watcher_safety.py`
   (REPO_DOCS + repo watcher sources) and add `worktrees/` to `.gitignore` so
   the runbook worktree convention leaves the shared checkout clean.

### Review Contract

- Acceptance criteria:
  - [ ] `docs/OVERNIGHT_ARC_WORKFLOW.md` states pre-flight, night loop,
        blocker channel, morning report, and the watcher exit states.
  - [ ] `scripts/watch_owned_pr.sh` passes bash -n and
        `scripts/audit_pr_watcher_safety.py` (no merge authority: no
        `gh pr merge`, no delete-branch, status-only messaging).
  - [ ] `AGENTS.md` 3c.2 and the `CLAUDE.md` compact-instructions bullet
        reference the runbook; edits are additive only.
  - [ ] MERGE-READY requires EVERY canonical required context (from
        `scripts/check_required_status_checks.py`, incl. diff-budget) present
        and success AND `claude-review` success AND 0 unresolved threads
        (fail closed on unfetched pages) AND no CHANGES_REQUESTED.
  - [ ] `scripts/audit_pr_watcher_safety.py` actually scans the new watcher
        and runbook (enrolled, not vacuous), with regression tests proving a
        repo watcher source containing a merge command FAILS the audit.
  - [ ] Required-context counting is app-pinned (GitHub Actions app id from
        `scripts/check_required_status_checks.py`) BEFORE latest-run
        selection, so a same-named check from another app can neither green
        the gate nor mask the genuine run.
- Reachability proof: process/docs + a standalone operator-run script; no
  runtime, API, UI, billing, or product surface. Proof is the rendered docs,
  the watcher-safety audit output, and a live one-cycle smoke of the script
  against an open PR.
- Affected surfaces: builder workflow contract (`AGENTS.md`, `CLAUDE.md`
  compact instructions), new docs file, new operator script.
- Risk areas: none at runtime. The watcher is status-only by construction and
  is covered by the existing watcher-safety audit; it holds no merge
  authority.
- Reviewer rules triggered: R1, R2, R10 (the watcher-safety audit is a gate
  predicate; its scan list is extended and the extension is itself exercised
  by the audit run in Verification).

### Files touched

- `.gitignore`
- `AGENTS.md`
- `CLAUDE.md`
- `docs/OVERNIGHT_ARC_WORKFLOW.md`
- `plans/PR-Overnight-Arc-Workflow.md`
- `scripts/audit_pr_watcher_safety.py`
- `scripts/watch_owned_pr.sh`
- `tests/test_audit_pr_watcher_safety.py`

## Mechanism

A new runbook doc; a new bash watcher script (derives repo root and origin,
reads the token from the repo `.env`, polls about every 29 minutes, exits on
the first actionable state); one additive AGENTS.md subsection; one additive
CLAUDE.md compact-instructions bullet. No code paths change.

## Intentional

- Repo-level codification (not session-local prose) so the workflow survives
  compaction -- repo contract docs are re-injected after compaction -- and
  reaches every session lineage. Machine-local session tooling mirrors it;
  the repo doc wins on conflict.
- The watcher checks the `claude-review` commit status in addition to
  required check-runs, matching the two-gate merge rule already in AGENTS
  3c.1.8 rather than the older single-gate shape.
- Task-agnostic by operator decision: no issue queue, no scheduler, no fixed
  task list is baked in; the task is chosen per night at pre-flight.

## Deferred

- Scheduled/cron dispatch of overnight arcs (operator spend decision; the
  manual kickoff prompt is the same artifact either way).
- Any arc queue or issue-template standardization (operator explicitly
  deferred task selection).

Parked hardening: none.

## Verification

- `scripts/audit_pr_watcher_safety.py` -- OK, and now actually scans the new
  watcher + runbook (enrolled in round 1; the pre-enrollment pass was
  vacuous).
- bash -n on `scripts/watch_owned_pr.sh` -- OK; live smoke against an open PR
  reports state correctly and exits ACTIONABLE on review threads + red
  required context (observed live on this PR's own round 1).
- `scripts/audit_plan_doc.py` on this plan -- OK.
- Round 3: required-context/app-pin extraction moved to the origin/main
  trusted ref; readiness blocks on unsettled required reruns; runbook gains
  the claude-review trust-boundary and wake-path pre-flight items; gate
  behavior tests waived to #2065 under the review cap.
- `git diff --name-status` -- two new files + additive edits to `AGENTS.md`,
  `CLAUDE.md`, `.gitignore`, and the audit scan list + this plan; no code
  paths.

## Estimated diff size

| File | LOC |
|---|---:|
| `.gitignore` | 3 |
| `AGENTS.md` | 18 |
| `CLAUDE.md` | 12 |
| `docs/OVERNIGHT_ARC_WORKFLOW.md` | 165 |
| `plans/PR-Overnight-Arc-Workflow.md` | 168 |
| `scripts/audit_pr_watcher_safety.py` | 9 |
| `scripts/watch_owned_pr.sh` | 101 |
| `tests/test_audit_pr_watcher_safety.py` | 27 |
| **Total** | **503** |

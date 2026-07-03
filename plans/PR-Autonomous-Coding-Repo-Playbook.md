# PR-Autonomous-Coding-Repo-Playbook

## Why this slice exists

Issue #1962 section 6 asks for a reusable autonomous-coding CI/CD playbook so
future repos can copy the working Atlas loop without inheriting every
Atlas-specific workflow. Earlier slices mapped the Atlas machine, documented the
long-running watcher handoff, measured runtime duplication, split push/review
attention from scheduled green confirmation, and added the monitoring spec.

This slice extracts the portable repo kit: which contracts to copy first, which
Atlas pieces are product-specific, what a smaller repo needs, and how to decide
when a repo is ready for long-running autonomous work.

## Scope (this PR)

Ownership lane: workflow/autonomous-ci-cd-map
Slice phase: Workflow/process

1. Add `docs/autonomous_coding_repo_playbook.md` with the reusable kit, bootstrap
   checklist, minimal CI architecture, long-running builder setup, review-loop
   pattern map, and replication test.
2. Link the playbook from the Atlas CI/CD map and long-running watcher handoff.
3. Archive the merged #1973 monitoring-spec plan as part of normal post-merge
   teardown.

### Review Contract

- Acceptance criteria:
  - [ ] The playbook separates portable contracts from Atlas-specific workflows.
  - [ ] The bootstrap checklist starts with process truth surfaces before
        product-specific CI.
  - [ ] The minimal CI architecture identifies required vs advisory/scheduled
        gates for a smaller repo.
  - [ ] The long-running builder setup preserves scheduled green confirmation,
        head-SHA ownership, clean worktree, and no active polling.
  - [ ] The review loop lists pattern classes that can become future rules,
        audits, tests, watcher changes, or hardening issues.
  - [ ] Existing #1962 docs link to the playbook without changing watcher
        runtime behavior.
- Reachability proof: docs-only surface; follow links from
  `docs/ci_cd_autonomous_coding_map.md` and
  `docs/long_running_session_watcher_handoff.md`.
- Affected surfaces: workflow docs / long-running session docs / plan archive.
- Risk areas: overclaiming portability, weakening scheduled merge semantics,
  stale plan archive state.
- Reviewer rules triggered: R1, R2, R10, R14.

### Files touched

- `docs/autonomous_coding_repo_playbook.md`
- `docs/ci_cd_autonomous_coding_map.md`
- `docs/long_running_session_watcher_handoff.md`
- `plans/INDEX.md`
- `plans/PR-Autonomous-Coding-Repo-Playbook.md`
- `plans/archive/PR-Long-Running-Monitoring-Spec.md`

## Mechanism

Add one Markdown playbook under `docs/` that translates the Atlas operating
model into reusable repo setup steps. The document keeps Atlas-specific checks
separate from repo-agnostic contracts and gives a smaller-repo CI stack so the
pattern can start lightweight.

Then add small cross-links from the two existing #1962 entrypoints. The merged
#1973 plan moves to `plans/archive/` so the active plan root represents the
current PR.

## Intentional

- No workflow, watcher executable, branch-protection, or local script behavior
  changes are included in this slice.
- The playbook recommends a minimum kit rather than a wholesale Atlas clone.
- The scheduled green-confirmation rule is preserved. The playbook does not
  treat push/review-event attention as merge authorization.

## Deferred

- #1962 section 7: public-facing Reddit/story version.
- Automation that generates a new-repo bootstrap PR from this playbook.
- Any repo-specific application of the playbook outside Atlas.

Parked hardening: none.

## Verification

- `grep -RInP "[^\x00-\x7F]" docs/autonomous_coding_repo_playbook.md docs/ci_cd_autonomous_coding_map.md docs/long_running_session_watcher_handoff.md plans/PR-Autonomous-Coding-Repo-Playbook.md || true` - passed, no non-ASCII output.
- `python scripts/archive_plans.py index` - passed, rebuilt `plans/INDEX.md` with the archived monitoring-spec plan.
- `python scripts/sync_pr_plan.py plans/PR-Autonomous-Coding-Repo-Playbook.md --check` - passed.
- `git diff --check` - passed.

## Estimated diff size

| File | LOC |
|---|---:|
| `docs/autonomous_coding_repo_playbook.md` | 191 |
| `docs/ci_cd_autonomous_coding_map.md` | 7 |
| `docs/long_running_session_watcher_handoff.md` | 7 |
| `plans/INDEX.md` | 3 |
| `plans/PR-Autonomous-Coding-Repo-Playbook.md` | 102 |
| `plans/archive/PR-Long-Running-Monitoring-Spec.md` | 0 |
| **Total** | **310** |

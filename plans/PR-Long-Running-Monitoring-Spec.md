# PR-Long-Running-Monitoring-Spec

## Why this slice exists

Issue #1962 section 5 asks for a monitoring/reporting spec for long-running
autonomous-coding arcs: metrics, data sources, per-arc pattern reports, and
failure classes worth codifying. The earlier slices mapped the CI/CD machine,
documented the long-running watcher, measured runtime duplication, and split the
push/review wake from the scheduled green-confirmation wake. What is still
missing is the lightweight reporting layer that turns each arc's red/green
loops and review misses into durable follow-up decisions instead of chat-only
memory.

This is not a runtime monitoring implementation. It defines the reporting
contract first so later automation can collect the same fields without inventing
new process semantics.

## Scope (this PR)

Ownership lane: workflow/autonomous-ci-cd-map
Slice phase: Workflow/process

1. Add the autonomous-coding arc monitoring/reporting spec: metric definitions,
   data sources, report cadence, report template, and codification thresholds.
2. Link the new reporting spec from the existing CI/CD map and long-running
   watcher handoff so future long-running sessions know where pattern drift is
   recorded.
3. Archive this session's merged #1972 hook-split plan as part of the normal
   post-merge teardown.

### Review Contract

- Acceptance criteria:
  - [ ] The spec names the issue #1962 metrics: PR cycle time, red-to-green
        loops, pushes per PR, recurring CI failures, review finding classes,
        stale branch count, and unresolved thread count.
  - [ ] The spec maps each metric to current data sources rather than requiring
        a new service: GitHub checks, review threads, PR bodies, plan docs,
        local audit output, watcher state, and maturity sweep baselines.
  - [ ] The spec includes a lightweight per-arc report format that can be filled
        manually today and automated later.
  - [ ] The spec lists failure classes worth codifying and gives promotion
        thresholds for turning repeats into AGENTS rules, audits, tests, or
        watcher changes.
  - [ ] Existing long-running docs point to the monitoring spec without
        changing watcher behavior or merge rules.
- Reachability proof: documentation surface only; reviewer can verify the linked
  entrypoints by following the links from `docs/ci_cd_autonomous_coding_map.md`
  and `docs/long_running_session_watcher_handoff.md` to the new spec.
- Affected surfaces: workflow docs / long-running session handoff / plan
  archive index.
- Risk areas: process drift, overclaiming automation that does not exist,
  stale plan archive state.
- Reviewer rules triggered: R1, R2, R10, R14.

### Files touched

- `docs/ci_cd_autonomous_coding_map.md`
- `docs/long_running_agent_monitoring_spec.md`
- `docs/long_running_session_watcher_handoff.md`
- `plans/INDEX.md`
- `plans/PR-Long-Running-Monitoring-Spec.md`
- `plans/archive/PR-Long-Running-Hook-Split.md`

## Mechanism

Add a new workflow doc under `docs/` that defines the per-arc monitoring
contract in stable Markdown sections:

- what to capture and when;
- metric definitions with source commands or artifacts;
- a per-arc report template;
- codification thresholds for repeated model/session failure classes;
- deferred automation boundaries.

Then add small cross-links from the CI/CD map and watcher handoff so a builder
or reviewer entering the long-running lane can find the reporting spec from the
existing docs. The merged #1972 plan is moved to `plans/archive/` and the plan
index is regenerated.

## Intentional

- No metrics collector, dashboard, workflow, or webhook is added in this slice.
  The point is to pin the report shape before implementing automation.
- The report format is Markdown-first. That keeps it fillable from the current
  GitHub CLI, watcher JSON, PR body, plan docs, and local audit output without
  adding a database or service.
- This PR links existing docs but does not change long-running watcher merge
  semantics. The scheduled 30-minute confirmation remains the merge source.

## Deferred

- Future issue #1962 slice: implement a collector or helper script that emits
  the report fields from `gh`, watcher state JSON, and local plan/audit files.
- Future issue #1962 slice: extract the reusable repo bootstrap/playbook once
  the Atlas-specific reporting contract has been reviewed.

Parked hardening: none.

## Verification

- `grep -RInP "[^\x00-\x7F]" docs/long_running_agent_monitoring_spec.md docs/ci_cd_autonomous_coding_map.md docs/long_running_session_watcher_handoff.md plans/PR-Long-Running-Monitoring-Spec.md || true` - passed, no non-ASCII output.
- `python scripts/sync_pr_plan.py plans/PR-Long-Running-Monitoring-Spec.md --check` - passed.
- `git diff --check` - passed.

## Estimated diff size

| File | LOC |
|---|---:|
| `docs/ci_cd_autonomous_coding_map.md` | 14 |
| `docs/long_running_agent_monitoring_spec.md` | 179 |
| `docs/long_running_session_watcher_handoff.md` | 13 |
| `plans/INDEX.md` | 3 |
| `plans/PR-Long-Running-Monitoring-Spec.md` | 116 |
| `plans/archive/PR-Long-Running-Hook-Split.md` | 0 |
| **Total** | **325** |

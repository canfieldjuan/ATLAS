# Autonomous Coding Repo Playbook

Issue: #1962

This playbook extracts the Atlas long-running coding system into a reusable
repo kit. It is meant for future repos that want the same small-slice,
PR-gated agent loop without copying every Atlas-specific workflow.

The core idea is simple: the model can write code, but the repo owns the truth.
Plans, tests, CI, review threads, and watcher state are the durable control
surfaces.

## What To Copy First

Start with the smallest kit that prevents silent drift:

1. `AGENTS.md` or equivalent repo contract.
2. A PR plan template with files touched, phase, scope, verification, and
   deferred work.
3. A local PR review wrapper that checks plan shape, diff size, dirty state,
   and PR body/reconciliation claims.
4. A PR body contract workflow.
5. A live review-thread reconciliation workflow.
6. Required secret scanning.
7. A session-state template for long-running work.
8. A documented watcher or timer path for green-confirmation checks.

Do not start with every Atlas package check. Start with the process gates that
make agents honest about what they changed.

## Portable Contracts

These rules are repo-agnostic:

- Every PR starts from a plan before implementation.
- The PR body names the plan and mirrors the current review/reconciliation
  state.
- The diff must match the plan's files touched and stay within the repo's slice
  budget unless the PR explicitly declares a budget override.
- Review comments are resolved only after the fix lands and the PR body agrees
  with live thread state.
- Long-running sessions own one lane and one active PR at a time.
- Merges require explicit operator approval or recorded standing merge
  authorization, plus clean CI, clean review threads, clean reconciliation,
  clean mergeability, matching expected head SHA, and a clean owned worktree.
- Red checks and review findings are fixed at the upstream cause, not with a
  symptom patch.
- Tests must prove behavior, including unhappy paths and edge cases.

These rules can live in `AGENTS.md`, `CLAUDE.md`, or a repo-local equivalent.

## Atlas-Specific Pieces

Do not copy these blindly:

- Atlas package workflows such as Content Ops, extracted packages, migrations,
  voice, invoicing, and Reddit listening.
- Atlas MCP inventory checks.
- Atlas-specific branch-protection context names.
- Atlas plan archive volume and historical index.
- Product-specific maturity baselines.
- Any local path under `/home/juan-canfield/Desktop/Atlas`.

Copy the pattern, then bind it to the new repo's real packages, tests, and
deployment surface.

## Bootstrap Checklist

Use this order for a new repo:

1. Add the agent contract.
   - Define plan-first work, small slices, ownership boundaries, dirty-tree
     rules, no destructive git, real-adapter test preference, and root-cause
     fixes.
2. Add the plan template.
   - Required sections: why, scope, review contract, files touched, mechanism,
     intentional, deferred, verification, estimated diff size.
3. Add a local review wrapper.
   - Start with low-dependency checks: dirty tree, plan exists, files touched
     match diff, diff size, PR body marker, and `git diff --check`.
4. Add PR body validation.
   - Use trusted-base workflow wiring and validator code if the PR can edit
     either surface.
5. Add review reconciliation.
   - Fail when the body claims automated-review findings are fixed or waived
     while unresolved bot threads still exist.
   - Run this from trusted-base code when the PR can edit the workflow or
     reconciliation script.
6. Add secret scanning.
   - Keep this required before other quality gates.
7. Add package-specific CI.
   - Enroll tests by path. Avoid a single expensive catch-all until the repo
     needs it.
8. Add branch-protection audit.
   - Verify the contexts you rely on are actually required.
9. Add long-running session state.
   - Keep it local and ignored. Store owned PR, expected head, standing merge
     authorization, hooks, and last watcher state.
10. Add watcher/timer support if long-running autonomous work is needed.
    - The watcher reports readiness only. Any standing-authorized merge happens
      in the builder/operator guard step after ownership, head, thread,
      reconciliation, mergeability, and worktree checks.
    - Closed PR watchers must shut down.
11. Add monitoring reports after several PRs.
    - Use the pattern report to decide which repeated misses become audits,
      tests, or stronger rules.

## Minimal CI Architecture

For a small repo, use this first:

| Layer | Required? | Purpose |
|---|---|---|
| PR body contract | Yes | Ensures every PR names a plan and uses the required structure. |
| Secret scan | Yes | Blocks the highest-risk mistake early; suppressions, baselines, and config must come from trusted-base code or an equivalent baseline-growth guard. |
| Unit tests | Yes | Runs the repo's normal test suite. |
| Plan/diff audit | Yes | Confirms scope, files touched, and diff budget match the plan; run the checker from trusted-base code when PRs can edit the audit workflow or script. |
| Review reconciliation | Yes, if bot review is used | Prevents stale "fixed" claims. |
| Dependency/security scan | Advisory or scheduled at first | Avoids slowing every slice before risk is known. |
| Maturity/brittleness scan | Advisory | Surfaces pattern drift without blocking early work. |
| Branch-protection audit | Scheduled | Confirms required checks stay required. |

Promote advisory checks to required only after they catch real failures that
would otherwise merge.

## Long-Running Builder Setup

Long-running agents need more structure than normal interactive PR work:

- one worktree per active slice;
- one watcher config per owned PR;
- one session-scoped state file updated after open, push, review fix, merge,
  and teardown;
- no active in-chat sleep loop for CI;
- no merge from a push/review attention wake;
- scheduled green confirmation before a standing-authorized merge;
- immediate stop on head-SHA mismatch or dirty owned worktree.

If there is no external push/review-event bridge, record it as unavailable. The
30-minute scheduled watcher is still useful, but the session does not have
immediate review wake-up coverage.

## Review Loop

The reviewer should not only list defects. They should classify patterns:

- fake or weak tests;
- plan weakening;
- stale PR body claims;
- scope drift;
- symptom patching;
- missing CI enrollment;
- stale generated fixtures;
- unsafe merge/readiness inference;
- watcher-state or teardown failures.

When a pattern repeats, record whether it belongs in the agent contract, a local
audit, a CI workflow, a test fixture, a watcher change, or a deferred hardening
issue.

## What Not To Build First

Avoid these until the basic loop has produced several PRs:

- dashboards;
- databases for process metrics;
- generalized multi-repo orchestration;
- auto-generated guards for imagined failures;
- broad required checks with high runtime and low signal;
- automatic review-thread resolution.

The first system should be boring, cheap, and hard to lie to.

## Replication Test

A repo is ready for long-running autonomous coding when a new builder can:

1. read the repo contract and plan template;
2. create a plan;
3. implement a small slice;
4. run local review;
5. open a PR with a contract-valid body;
6. receive review or CI feedback;
7. fix the upstream cause;
8. reconcile the PR body with live review state;
9. merge only with explicit operator approval or recorded standing merge
   authorization, and after green checks, clean ownership state, expected head
   SHA, clean worktree, clean review threads, clean reconciliation, and clean
   mergeability;
10. shut down the completed PR's watcher/timer, tear down the worktree, and
    continue the next approved slice.

If any step depends on tribal knowledge, add it to the repo contract before
scaling the loop.

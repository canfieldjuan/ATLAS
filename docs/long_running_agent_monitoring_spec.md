# Long-Running Agent Monitoring Spec

Issue: #1962

This spec defines the lightweight reporting layer for long-running
autonomous-coding arcs. It answers: what should the builder/reviewer/operator
capture from each arc, where does the evidence come from today, and when does a
repeated failure pattern become a durable rule, audit, test, or watcher change?

Monitoring here does not mean a new service. It means a read-only report built
from GitHub, plan docs, watcher state, local audit output, and existing CI
baselines. The report can be filled manually now and automated later.

## Principles

1. Current artifacts beat memory. Metrics cite GitHub, repo docs, watcher JSON,
   local audit output, or CI logs.
2. Collection is read-only. A report never merges, closes, rebases, resolves
   threads, or changes branch protection.
3. PR-level events roll up into arc-level patterns. A single red run may be
   noise; repeated red classes are process data.
4. Codify repeats. If the same failure shape costs review or CI time twice, the
   report should name whether it belongs in `AGENTS.md`, an audit script, a
   test, a watcher rule, or `HARDENING.md`.
5. No active polling. Reports use scheduled watcher snapshots, operator/review
   wakes, and one-shot `gh` reads during an active turn.

## Cadence

| Moment | Capture | Owner |
|---|---|---|
| PR opened | PR number, branch, head SHA, plan, watcher session id, first check set. | Builder |
| Push/review wake | New head SHA, actionable findings, red checks, fixed threads, PR body updates. | Builder while fixing; reviewer while judging |
| Scheduled watcher wake | Watcher state, pending/red/ready status, mergeability, reconciliation state. | Builder |
| PR merged | Cycle time, push count, red-to-green loops, final head, teardown result. | Builder |
| Arc checkpoint | Pattern classes, repeated causes, codification decisions, deferred hardening. | Builder + reviewer/operator |

Arc checkpoints should happen at the end of the issue, after five PRs in one
arc, or earlier when a BLOCKER/MAJOR exposes a repeat class.

## Metric Definitions

| Metric | Definition | Current source | Why it matters |
|---|---|---|---|
| PR cycle time | `mergedAt - createdAt`; for open PRs use current time as the temporary end. | `gh pr view <pr> --json createdAt,mergedAt` | Shows how long a slice occupies review/CI capacity. |
| Red-to-green loops | Count distinct red/check-failed or actionable-review states that require a follow-up push before green. | `gh pr checks <pr>`, `gh run view`, review thread reads, watcher state JSON | Finds repeated classes that local review or planning should catch earlier. |
| Pushes per PR | Number of commits/head updates after PR open, plus the final merge commit separately. | `gh pr view <pr> --json commits`, PR timeline when needed | High counts can mean weak plan scope, missed callers, or noisy tests. |
| Recurring CI failures | Failed check name plus normalized class, such as test collection, lint, contract, reconciliation, or infra. | `gh pr checks <pr>` and failing `gh run view <run-id> --log` excerpts | Distinguishes product defects from workflow friction. |
| Review finding classes | Human/Codex/Copilot findings grouped by verdict, rule ID, and root class. | Review threads, PR review bodies, live reconciliation output | Converts review misses into durable prevention work. |
| Stale branch count | Open PRs in the arc with head/base drift, head SHA mismatch, or merge state not clean. | `gh pr list --state open`, `gh pr view --json mergeStateStatus,headRefOid` | Detects branch/session drift before a builder overwrites or merges the wrong work. |
| Unresolved thread count | Unresolved review threads on the current head, split by bot/human where useful. | GraphQL review threads and `live-reconciliation` | Prevents false-green PR bodies and premature merges. |
| Local-audit misses | Findings that GitHub caught but `scripts/local_pr_review.sh` did not. | Local review output, PR checks, reviewer notes | Identifies candidates for new local audits or plan-contract rules. |
| Maturity baseline drift | Ratchet or maturity-sweep movement caused by the slice. | Maturity sweep output and baselines | Shows when a slice moves a quality threshold or trips a non-product guard. |

## Data Sources

Use one-shot reads while actively handling a PR or compiling an arc report:

```bash
gh pr view <pr> --repo canfieldjuan/ATLAS \
  --json number,title,url,state,createdAt,mergedAt,headRefName,headRefOid,baseRefName,mergeStateStatus,reviewDecision,commits,body

gh pr checks <pr> --repo canfieldjuan/ATLAS
gh run view <run-id> --repo canfieldjuan/ATLAS --json jobs
```

Thread-level state comes from the same GraphQL review-thread reads used by
live-reconciliation. The report needs the unresolved count, author class, rule
or verdict when present, and whether a later push fixed or waived it.

Local inputs:

- `plans/PR-<Slice>.md` and the mirrored PR body.
- `SESSION_STATE.local.md` for lane ownership, watcher hooks, standing
  authorization, last safe action, and fix-mode baton.
- `~/.local/state/atlas-pr-watchers/<session-id>.json` for scheduled watcher
  state.
- `scripts/local_pr_review.sh` or `scripts/push_pr.sh` output when it caught or
  missed a class.
- CI logs and maturity sweep output only as short excerpts. Do not paste
  secrets, headers, env values, or full logs into reports.

## Per-Arc Report Format

Create one report per long-running arc when the arc ends or reaches a checkpoint.
Suggested path for future reports:

```text
docs/autonomous-coding-reports/YYYY-MM-DD-<arc-slug>.md
```

Template:

```md
# Autonomous Coding Arc Report: <arc>

Issue: #<issue>
Window: <start> to <end>
Owner/session: <builder/reviewer labels>
Scope: <one-sentence lane>

## Slices

| PR | Plan | Phase | Final head | Result |
|---|---|---|---|---|
| #123 | plans/PR-Example.md | Workflow/process | abc123 | merged |

## Metrics

| Metric | Value | Evidence |
|---|---:|---|
| PR cycle time median | <duration> | <gh command or PR links> |
| Red-to-green loops | <count> | <check/thread evidence> |
| Pushes per PR median | <count> | <commit/head evidence> |
| Recurring CI failures | <count by class> | <check names> |
| Review finding classes | <count by class> | <thread/verdict evidence> |
| Stale branch count | <count> | <mergeState/head evidence> |
| Unresolved thread count at merge | <count> | <live-reconciliation/thread evidence> |

## Red/Green Loop Ledger

| PR | Signal | Class | Root cause | Fix | Prevention candidate |
|---|---|---|---|---|---|
| #123 | check failed | CI enrollment | test script not added to workflow | added workflow step | AGENTS/audit |

## Review Finding Classes

| Class | Seen | Severity | Example PRs | Codification decision |
|---|---:|---|---|---|
| fake adapter | 2 | MAJOR | #A, #B | AGENTS rule + real-adapter test |

## Codification Backlog

| Pattern | Promote to | Next action | Owner |
|---|---|---|---|
| <pattern> | AGENTS / audit / test / watcher / HARDENING | <next slice or reason deferred> | <owner> |

## Deferred Or Waived

- <item> -- <why not codified now and what would reopen it>
```

## Failure Classes Worth Codifying

| Failure class | Codify as | Typical trigger |
|---|---|---|
| Missing CI enrollment | Workflow audit or AGENTS rule | New test/script passes locally but is absent from the workflow run list. |
| Fake adapters or mock call-arg tests | AGENTS test rule plus real-adapter regression | Test proves a fake's call shape instead of observable adapter state. |
| Stale generated fixtures | Generator-backed fixture check | Hand-authored contract artifact drifts from the producer. |
| Stale PR body claims | PR body/live-reconciliation check | Body says AI findings are fixed while current threads remain open. |
| Unresolved AI threads | Live reconciliation or reviewer checklist | Bot/human thread remains unresolved at merge time. |
| Branch/head drift | Session ownership guard or watcher head mismatch stop | Remote head changes unexpectedly or branch is behind base. |
| Scope/lane drift | `SESSION_STATE.local.md` ownership rule or drift audit | Builder touches a PR or lane not assigned to the session. |
| Push/review hook confusion | Watcher docs or event-bridge guard | Notification is recorded as a builder wake hook, or event wake is treated as merge authorization. |
| Active polling | AGENTS/watcher rule | Builder waits in-chat for CI instead of relying on hooks. |

## Promotion Thresholds

Promote immediately when the class affects security, billing, data deletion,
customer-visible truth, CI false-green behavior, or merge ownership.

For other classes:

- One isolated, non-blocking miss: record in the report or `HARDENING.md`.
- Two misses in one arc: add an AGENTS/reviewer-rule note or a named follow-up
  issue.
- Two misses that are machine-checkable: add or plan an audit script/test.
- Any miss that local review should have caught: decide whether to extend
  `scripts/local_pr_review.sh`, `pre_push_audit.sh`, or the plan/PR body
  contract.
- Any watcher-state miss: update watcher docs first; add automation only after
  the allowed merge source remains explicit.

## Deferred Automation

Later automation can generate this report from `gh`, watcher JSON, plan docs,
and local audit output. It should stay read-only unless a separate PR explicitly
adds a safe enforcement gate. Do not turn the report into a merge authority; the
scheduled watcher and AGENTS merge guards remain the source of merge readiness.

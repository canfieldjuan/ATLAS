# PR-Agents-Session-PR-Map

## Why this slice exists

Builder sessions have drifted after context compaction: resuming in the wrong
lane, merging PRs that were not created by the current task, and editing code
owned by an open PR from another session. The existing bootstrap says to check
open PRs, but it does not give the builder a persistent ownership ledger to
distinguish "mine" from "not mine" after compaction.

This slice adds a required local session ownership map. It makes PR ownership a
first-class checkpoint before merge, comment handling, or new-code work.

## Scope (this PR)

Ownership lane: repo-workflow/session-discipline
Slice phase: Workflow/process

1. Add a durable template for a local `SESSION_STATE.local.md` ownership map.
2. Ignore the local map so volatile PR state does not churn in commits.
3. Update `AGENTS.md` so builders must read/update the map and treat unlisted
   PRs as not theirs.
4. Update the fresh-session and drift-redirect bootstrap prompts to require the
   map before PR actions.
5. Teach the plan/code consistency audit to skip path claims that are
   intentionally ignored by git, so local-only artifacts do not pass locally and
   fail in CI.

### Files touched

- `plans/PR-Agents-Session-PR-Map.md`
- `.gitignore`
- `AGENTS.md`
- `docs/SESSION_BOOTSTRAP.md`
- `docs/SESSION_STATE_TEMPLATE.md`
- `scripts/audit_plan_code_consistency.py`
- `tests/test_audit_plan_code_consistency.py`

## Mechanism

`docs/SESSION_STATE_TEMPLATE.md` defines the local state shape:

- current lane and task
- owned active PR, branch, plan, and expected head
- merged PRs from this session
- explicit "not mine" open PRs
- last safe action and resume checklist

`AGENTS.md` makes that map mandatory before PR mutation. If the current PR is
absent from the owned slot, or the map is missing/stale after compaction, the
builder must stop and ask instead of inferring ownership from lane proximity.

The plan/code consistency audit now checks git-ignore status before reporting a
missing path claim. That keeps deliberately local artifacts like the session map
from producing a local-green/CI-red split.

## Intentional

- The actual `SESSION_STATE.local.md` file is ignored. It is a local continuity
  artifact, not repo truth.
- This does not add an automated GitHub gate. The immediate problem is agent
  behavior after compaction; the contract and template give the session a
  stable object to re-read.
- Existing open PR #1188 is not touched. It is explicitly outside this process
  slice.
- Ignored path claims are not treated as missing because they cannot exist in
  the clean CI checkout by design.

## Deferred

- Parked hardening: none.
- A future tooling slice can add a script that checks `SESSION_STATE.local.md`
  against `gh pr list` and fails before merge when ownership is ambiguous.

## Verification

- `python -m pytest tests/test_audit_plan_code_consistency.py -q`
- Local PR review bundle with the PR body file supplied.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan doc | 89 |
| AGENTS contract | 35 |
| Bootstrap prompt | 17 |
| Template + ignore | 71 |
| Audit skip + test | 36 |
| **Total** | **248** |

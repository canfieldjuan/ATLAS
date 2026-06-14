# PR-Spark-Routing-Light-Enforcement

## Why this slice exists

#1543 codified Spark as the preferred lightweight scout for bounded read-only
work, but the policy still depends on memory. The operator wants light
enforcement, not a hard CI gate. This slice makes the routing decision visible
in the mandatory local session-state handoff so future sessions record whether
Spark or another subagent was used, considered, or not applicable.

## Scope (this PR)

Ownership lane: workflow/process
Slice phase: Workflow/process

1. Add a `Spark/subagent routing` field to the session-state template.
2. Update the builder workflow and fresh-session bootstrap to require filling
   that field when maintaining the session map.
3. Keep enforcement lightweight: no new scripts, CI gates, or local-review
   blockers.

### Files touched

- `AGENTS.md`
- `docs/SESSION_BOOTSTRAP.md`
- `docs/SESSION_STATE_TEMPLATE.md`
- `plans/PR-Spark-Routing-Light-Enforcement.md`

### Review Contract

Acceptance criteria:

- The session-state template has a visible Spark/subagent routing field.
- `AGENTS.md` says the session map must record Spark/subagent routing used or
  considered.
- The fresh-session bootstrap tells new sessions to fill in the routing field.
- The slice does not add hard enforcement through CI, local-review scripts, or
  PR mutation gates.

Affected surfaces:

- Atlas workflow/process documentation and local session-state template only.

Risk areas:

- Wording must not imply subagents can own judgment, edits, Git/GitHub
  mutations, or final synthesis.
- The field should support "not applicable" so tiny direct tasks do not produce
  noisy process churn.

Reviewer rules triggered: R1, R12, R14

## Mechanism

The mandatory local session map gets a single new top-level line:
`Spark/subagent routing: used ... | considered ... | not applicable`.
`AGENTS.md` and the fresh-session bootstrap both name that field in the list of
session-state data builders must maintain. This creates lightweight
accountability in the handoff state without adding a mechanical gate.

## Intentional

- No new checker or CI gate. A subjective "should Spark have been used" audit
  would be noisy as a hard blocker.
- No changes to the PR body contract. The routing note belongs in local session
  state, not every PR description.
- The field allows "not applicable" for small direct commands and edit-target
  reads where main should work directly.

## Deferred

- A script-backed checker remains deferred unless repeated misses show that the
  session-state field is insufficient.

Parked hardening: none.

## Verification

- Doc diff inspection: passed.
- `python scripts/sync_pr_plan.py plans/PR-Spark-Routing-Light-Enforcement.md --check`: passed.
- Body-aware local PR review: passed.

## Estimated diff size

| File | LOC |
|---|---:|
| `AGENTS.md` | 7 |
| `docs/SESSION_BOOTSTRAP.md` | 2 |
| `docs/SESSION_STATE_TEMPLATE.md` | 1 |
| `plans/PR-Spark-Routing-Light-Enforcement.md` | 91 |
| **Total** | **101** |

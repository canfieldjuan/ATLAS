# In-Flight PRs

Last updated: 2026-05-17T21:01Z by codex-2026-05-17

Add a row before opening a PR (session protocol step 2). Drop the row when the PR merges (step 4). See [`../COORDINATION.md`](../COORDINATION.md) for protocol details.

| PR | Title | Touches | Owner | Don't conflict with |
|---|---|---|---|---|
| #576 | Post-575 extraction state closeout | `docs/extraction/coordination/inflight.md`, `docs/extraction/coordination/state.md`, `docs/extraction/reasoning_core_current_state_audit_2026-05-17.md`, `plans/PR-Post-575-Extraction-State-Closeout.md` | codex-2026-05-17 | Avoid editing extraction coordination/state or reasoning-core current-state audit |

This table is for PRs we need to coordinate around, not a mirror of `gh pr list`. Use `gh pr list --state open` for the full inventory.

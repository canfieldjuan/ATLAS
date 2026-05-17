# In-Flight PRs

Last updated: 2026-05-17T19:46Z by codex-2026-05-17

Add a row before opening a PR (session protocol step 2). Drop the row when the PR merges (step 4). See [`../COORDINATION.md`](../COORDINATION.md) for protocol details.

| PR | Title | Touches | Owner | Don't conflict with |
|---|---|---|---|---|
| TBD | Document reasoning graph boundary closeout | `docs/extraction/reasoning_core_current_state_audit_2026-05-17.md`, `docs/extraction/coordination/inflight.md`, `docs/extraction/coordination/queue.md`, `docs/extraction/coordination/state.md`, `docs/extraction/coordination/decisions.md`, `plans/PR-Reasoning-Core-Graph-Boundary-Closeout-2026-05-17.md` | codex-2026-05-17 | Avoid changing the reasoning-core graph/reflection boundary docs until this lands. |

This table is for PRs we need to coordinate around, not a mirror of `gh pr list`. Use `gh pr list --state open` for the full inventory.

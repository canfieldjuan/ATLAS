# In-Flight PRs

Last updated: 2026-05-17T00:53Z by codex-2026-05-16

Add a row before opening a PR (session protocol step 2). Drop the row when the PR merges (step 4). See [`../COORDINATION.md`](../COORDINATION.md) for protocol details.

| PR | Title | Touches | Owner | Don't conflict with |
|---|---|---|---|---|
| #563 | Reconcile extracted reasoning core current state | `docs/extraction/coordination/inflight.md`, `docs/extraction/coordination/state.md`, `docs/extraction/coordination/queue.md`, `docs/extraction/reasoning_core_current_state_audit_2026-05-17.md`, `docs/extraction/reasoning_boundary_audit_2026-05-03.md`, `plans/PR-Reasoning-Core-Current-State.md` | codex-2026-05-16 | Avoid editing extracted_reasoning_core status, queue, or next-slice recommendation while this reconciles stale reasoning-core docs. |

This table is for PRs we need to coordinate around, not a mirror of `gh pr list`. Use `gh pr list --state open` for the full inventory.

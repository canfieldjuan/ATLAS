# In-Flight PRs

Last updated: 2026-05-17T19:00Z by codex-2026-05-17

Add a row before opening a PR (session protocol step 2). Drop the row when the PR merges (step 4). See [`../COORDINATION.md`](../COORDINATION.md) for protocol details.

| PR | Title | Touches | Owner | Don't conflict with |
|---|---|---|---|---|
| TBD | Repair Atlas graph routing tests for port adapter seam | `docs/extraction/coordination/inflight.md`, `docs/extraction/coordination/queue.md`, `docs/extraction/coordination/state.md`, `plans/PR-Reasoning-Core-Graph-Routing-Test-Seam-2026-05-17.md`, `tests/test_reasoning_graph_routing.py`, `tests/test_reasoning_graph_summary.py` | codex-2026-05-17 | Avoid editing Atlas graph-routing tests or reasoning-core coordination state until this PR lands. |

This table is for PRs we need to coordinate around, not a mirror of `gh pr list`. Use `gh pr list --state open` for the full inventory.

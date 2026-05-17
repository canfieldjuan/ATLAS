# In-Flight PRs

Last updated: 2026-05-17T18:12Z by codex-2026-05-17

Add a row before opening a PR (session protocol step 2). Drop the row when the PR merges (step 4). See [`../COORDINATION.md`](../COORDINATION.md) for protocol details.

| PR | Title | Touches | Owner | Don't conflict with |
|---|---|---|---|---|
| #570 | Promote reasoning graph reason node into core | `atlas_brain/reasoning/graph.py`, `docs/extraction/coordination/inflight.md`, `docs/extraction/coordination/queue.md`, `docs/extraction/coordination/state.md`, `extracted_reasoning_core/graph_nodes.py`, `plans/PR-Reasoning-Core-Node-Reason-2026-05-17.md`, `tests/test_extracted_reasoning_core_graph_nodes.py` | codex-2026-05-17 | Avoid editing reasoning graph node execution or graph-state coordination wording until this PR lands. |

This table is for PRs we need to coordinate around, not a mirror of `gh pr list`. Use `gh pr list --state open` for the full inventory.

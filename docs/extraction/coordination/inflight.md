# In-Flight PRs

Last updated: 2026-05-17T19:31Z by codex-2026-05-17

Add a row before opening a PR (session protocol step 2). Drop the row when the PR merges (step 4). See [`../COORDINATION.md`](../COORDINATION.md) for protocol details.

| PR | Title | Touches | Owner | Don't conflict with |
|---|---|---|---|---|
| TBD | Route reflection LLM analysis through port adapter | `atlas_brain/reasoning/graph.py`, `atlas_brain/reasoning/reflection.py`, `docs/extraction/coordination/inflight.md`, `docs/extraction/coordination/queue.md`, `docs/extraction/coordination/state.md`, `plans/PR-Reasoning-Core-Reflection-Port-Adapter-2026-05-17.md`, `tests/test_anthropic_timeout.py`, `tests/test_atlas_reasoning_reflection_tracing.py`, `tests/test_reasoning_graph_routing.py` | codex-2026-05-17 | Avoid editing reflection LLM routing, graph legacy helper cleanup, or reasoning-core coordination state until this PR lands. |

This table is for PRs we need to coordinate around, not a mirror of `gh pr list`. Use `gh pr list --state open` for the full inventory.

# In-Flight PRs

Last updated: 2026-05-14T19:35Z by codex-2026-05-14-content-ops-intervention-reasoning

Add a row before opening a PR (session protocol step 2). Drop the row when the PR merges (step 4). See [`../COORDINATION.md`](../COORDINATION.md) for protocol details.

| PR | Title | Touches | Owner | Don't conflict with |
|---|---|---|---|---|
| (in flight) | Add Content Ops intervention reasoning provider | NEW: `plans/PR-Content-Ops-Intervention-Reasoning-Provider.md`. EDIT: `atlas_brain/_content_ops_reasoning.py`; `tests/test_atlas_content_ops_reasoning.py`; `docs/audits/ai_content_ops_deferred_backlog_2026-05-11.md`; `docs/extraction/coordination/inflight.md`. | codex-2026-05-14-content-ops-intervention-reasoning | frontend Content Ops UI slices; extracted reasoning core slices |

This table is for PRs we need to coordinate around, not a mirror of `gh pr list`. Use `gh pr list --state open` for the full inventory.

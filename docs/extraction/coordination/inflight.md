# In-Flight PRs

Last updated: 2026-05-17T22:41Z by codex-2026-05-17

Add a row before opening a PR (session protocol step 2). Drop the row when the PR merges (step 4). See [`../COORDINATION.md`](../COORDINATION.md) for protocol details.

| PR | Title | Touches | Owner | Don't conflict with |
|---|---|---|---|---|
| #581 | Content Ops ingestion inspect UI | `atlas-intel-ui/src/pages/ContentOpsNewRun.tsx`, `docs/frontend/content_ops_frontend_contract.md`, `docs/extraction/coordination/inflight.md`, `docs/extraction/coordination/state.md`, `plans/PR-Content-Ops-Ingestion-Inspect-UI.md` | codex-2026-05-17 | Avoid editing the Content Ops new-run UI and ingestion-inspect UI docs |

This table is for PRs we need to coordinate around, not a mirror of `gh pr list`. Use `gh pr list --state open` for the full inventory.

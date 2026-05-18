# In-Flight PRs

Last updated: 2026-05-18T14:49Z by codex-2026-05-17

Add a row before opening a PR (session protocol step 2). Drop the row when the PR merges (step 4). See [`../COORDINATION.md`](../COORDINATION.md) for protocol details.

| PR | Title | Touches | Owner | Don't conflict with |
|---|---|---|---|---|
| pending | Content Ops ingestion import diagnostics UI | `atlas-intel-ui/src/api/contentOps.ts`, `atlas-intel-ui/src/pages/ContentOpsNewRun.tsx`, `docs/frontend/content_ops_frontend_contract.md`, `docs/extraction/coordination/inflight.md`, `docs/extraction/coordination/state.md`, `plans/PR-Content-Ops-Ingestion-Import-Error-UI.md` | codex-2026-05-17 | Avoid editing Content Ops frontend ingestion import error handling |

This table is for PRs we need to coordinate around, not a mirror of `gh pr list`. Use `gh pr list --state open` for the full inventory.

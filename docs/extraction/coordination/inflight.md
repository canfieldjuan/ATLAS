# In-Flight PRs

Last updated: 2026-05-17T22:33Z by codex-2026-05-17

Add a row before opening a PR (session protocol step 2). Drop the row when the PR merges (step 4). See [`../COORDINATION.md`](../COORDINATION.md) for protocol details.

| PR | Title | Touches | Owner | Don't conflict with |
|---|---|---|---|---|
| #580 | Content Ops ingestion inspect frontend contract | `atlas-intel-ui/src/api/contentOps.ts`, `atlas-intel-ui/src/api/contentOps.contract.ts`, `atlas-intel-ui/src/api/__fixtures__/contentOps/ingestion-inspect.json`, `atlas-intel-ui/src/domain/contentOps/types.ts`, `atlas-intel-ui/src/domain/contentOps/fromWire.ts`, `atlas-intel-ui/src/domain/contentOps/contract.ts`, `atlas-intel-ui/src/domain/contentOps/index.ts`, `docs/frontend/content_ops_frontend_contract.md`, `docs/extraction/coordination/inflight.md`, `docs/extraction/coordination/state.md`, `plans/PR-Content-Ops-Ingestion-Inspect-Frontend-Contract.md` | codex-2026-05-17 | Avoid editing Content Ops frontend API/domain contract files |

This table is for PRs we need to coordinate around, not a mirror of `gh pr list`. Use `gh pr list --state open` for the full inventory.

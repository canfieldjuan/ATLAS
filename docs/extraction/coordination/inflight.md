# In-Flight PRs

Last updated: 2026-05-15T23:05Z by codex-2026-05-15

Add a row before opening a PR (session protocol step 2). Drop the row when the PR merges (step 4). See [`../COORDINATION.md`](../COORDINATION.md) for protocol details.

| PR | Title | Touches | Owner | Don't conflict with |
|---|---|---|---|---|
| draft | Content Ops reasoning status UI parity | `atlas-intel-ui/src/api/contentOps.ts`, `atlas-intel-ui/src/domain/contentOps/types.ts`, `atlas-intel-ui/src/domain/contentOps/fromWire.ts`, `atlas-intel-ui/src/pages/ContentOpsNewRun.tsx`, `docs/extraction/coordination/inflight.md`, `plans/PR-Content-Ops-Reasoning-Status-UI-Parity.md` | codex-2026-05-15 | Avoid Content Ops frontend reasoning status rendering/types |

This table is for PRs we need to coordinate around, not a mirror of `gh pr list`. Use `gh pr list --state open` for the full inventory.

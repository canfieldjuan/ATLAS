# In-Flight PRs

Last updated: 2026-05-15T18:46Z by codex-2026-05-15

Add a row before opening a PR (session protocol step 2). Drop the row when the PR merges (step 4). See [`../COORDINATION.md`](../COORDINATION.md) for protocol details.

| PR | Title | Touches | Owner | Don't conflict with |
|---|---|---|---|---|
| draft | Content Ops generated asset preview UX | `atlas-intel-ui/src/api/contentOps.ts`, `atlas-intel-ui/src/pages/ContentOpsAssetsReview.tsx`, `docs/extraction/coordination/inflight.md`, `plans/PR-Content-Ops-Generated-Asset-Preview-UX.md` | codex-2026-05-15 | Avoid generated asset review UI |

This table is for PRs we need to coordinate around, not a mirror of `gh pr list`. Use `gh pr list --state open` for the full inventory.

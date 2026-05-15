# In-Flight PRs

Last updated: 2026-05-15T02:41Z by codex-2026-05-15-content-ops-asset-preview

Add a row before opening a PR (session protocol step 2). Drop the row when the PR merges (step 4). See [`../COORDINATION.md`](../COORDINATION.md) for protocol details.

| PR | Title | Touches | Owner | Don't conflict with |
|---|---|---|---|---|
| draft | Content Ops asset previews | `atlas-intel-ui/src/api/contentOps.ts`, `atlas-intel-ui/src/pages/ContentOpsAssetsReview.tsx` | codex-2026-05-15-content-ops-asset-preview | Avoid generated asset review UI/API-type edits |

This table is for PRs we need to coordinate around, not a mirror of `gh pr list`. Use `gh pr list --state open` for the full inventory.

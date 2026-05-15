# In-Flight PRs

Last updated: 2026-05-15T02:18Z by codex-2026-05-15-content-assets-review-row-ids

Add a row before opening a PR (session protocol step 2). Drop the row when the PR merges (step 4). See [`../COORDINATION.md`](../COORDINATION.md) for protocol details.

| PR | Title | Touches | Owner | Don't conflict with |
|---|---|---|---|---|
| draft | Content Ops asset review row ids | generated asset ports, Postgres adapters, export helpers, asset API tests | codex-2026-05-15-content-assets-review-row-ids | Avoid generated-asset list/export row-shape edits |

This table is for PRs we need to coordinate around, not a mirror of `gh pr list`. Use `gh pr list --state open` for the full inventory.

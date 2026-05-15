# In-Flight PRs

Last updated: 2026-05-15T02:31Z by codex-2026-05-15-content-ops-asset-bulk-review

Add a row before opening a PR (session protocol step 2). Drop the row when the PR merges (step 4). See [`../COORDINATION.md`](../COORDINATION.md) for protocol details.

| PR | Title | Touches | Owner | Don't conflict with |
|---|---|---|---|---|
| draft | Content Ops asset bulk review | `extracted_content_pipeline/api/generated_assets.py`, `tests/test_extracted_content_asset_api.py`, `atlas-intel-ui/src/api/contentOps.ts`, `atlas-intel-ui/src/pages/ContentOpsAssetsReview.tsx` | codex-2026-05-15-content-ops-asset-bulk-review | Avoid generated asset review API/UI edits |

This table is for PRs we need to coordinate around, not a mirror of `gh pr list`. Use `gh pr list --state open` for the full inventory.

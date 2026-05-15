# In-Flight PRs

Last updated: 2026-05-15T02:14Z by codex-2026-05-15-content-ops-reasoning-upsert

Add a row before opening a PR (session protocol step 2). Drop the row when the PR merges (step 4). See [`../COORDINATION.md`](../COORDINATION.md) for protocol details.

| PR | Title | Touches | Owner | Don't conflict with |
|---|---|---|---|---|
| draft | Content Ops reasoning context upsert | `extracted_content_pipeline/campaign_reasoning_postgres.py`, `extracted_content_pipeline/storage/migrations/278_campaign_reasoning_context_upsert.sql`, `extracted_content_pipeline/manifest.json`, `tests/test_extracted_campaign_reasoning_postgres.py`, `tests/test_extracted_campaign_manifest.py` | codex-2026-05-15-content-ops-reasoning-upsert | Avoid campaign reasoning context repository/migration edits |

This table is for PRs we need to coordinate around, not a mirror of `gh pr list`. Use `gh pr list --state open` for the full inventory.

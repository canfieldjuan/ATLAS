# In-Flight PRs

Last updated: 2026-05-15T01:12Z by codex-2026-05-15-content-ops-db-reasoning-hardening

Add a row before opening a PR (session protocol step 2). Drop the row when the PR merges (step 4). See [`../COORDINATION.md`](../COORDINATION.md) for protocol details.

| PR | Title | Touches | Owner | Don't conflict with |
|---|---|---|---|---|
| draft | Content Ops DB reasoning hardening | `atlas_brain/_content_ops_reasoning.py`, `atlas_brain/config.py`, `extracted_content_pipeline/campaign_reasoning_postgres.py`, reasoning provider tests | codex-2026-05-15-content-ops-db-reasoning-hardening | Avoid DB reasoning provider/storage edits |

This table is for PRs we need to coordinate around, not a mirror of `gh pr list`. Use `gh pr list --state open` for the full inventory.

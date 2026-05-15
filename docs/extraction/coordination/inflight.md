# In-Flight PRs

Last updated: 2026-05-15T02:55Z by codex-2026-05-15-content-ops-db-reasoning-sweeper

Add a row before opening a PR (session protocol step 2). Drop the row when the PR merges (step 4). See [`../COORDINATION.md`](../COORDINATION.md) for protocol details.

| PR | Title | Touches | Owner | Don't conflict with |
|---|---|---|---|---|
| draft | Content Ops DB reasoning sweeper | `extracted_content_pipeline/campaign_reasoning_postgres.py`, `scripts/cleanup_extracted_campaign_reasoning_contexts.py`, `tests/test_extracted_campaign_reasoning_postgres.py`, `tests/test_extracted_campaign_reasoning_cleanup_cli.py` | codex-2026-05-15-content-ops-db-reasoning-sweeper | Avoid DB reasoning repository cleanup and host cleanup CLI edits |

This table is for PRs we need to coordinate around, not a mirror of `gh pr list`. Use `gh pr list --state open` for the full inventory.

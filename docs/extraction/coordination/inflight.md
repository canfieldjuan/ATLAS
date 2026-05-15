# In-Flight PRs

Last updated: 2026-05-15T17:32Z by codex-2026-05-15

Add a row before opening a PR (session protocol step 2). Drop the row when the PR merges (step 4). See [`../COORDINATION.md`](../COORDINATION.md) for protocol details.

| PR | Title | Touches | Owner | Don't conflict with |
|---|---|---|---|---|
| draft | Content Ops reasoning upsert validation | `scripts/upsert_extracted_campaign_reasoning_contexts.py`, `tests/test_extracted_campaign_reasoning_upsert_cli.py`, `extracted_content_pipeline/README.md`, `extracted_content_pipeline/docs/host_install_runbook.md`, `extracted_content_pipeline/STATUS.md`, `docs/audits/ai_content_ops_deferred_backlog_2026-05-11.md` | codex-2026-05-15 | Avoid DB reasoning upsert/admin edits |

This table is for PRs we need to coordinate around, not a mirror of `gh pr list`. Use `gh pr list --state open` for the full inventory.

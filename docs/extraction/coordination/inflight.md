# In-Flight PRs

Last updated: 2026-05-15T18:11Z by codex-2026-05-15

Add a row before opening a PR (session protocol step 2). Drop the row when the PR merges (step 4). See [`../COORDINATION.md`](../COORDINATION.md) for protocol details.

| PR | Title | Touches | Owner | Don't conflict with |
|---|---|---|---|---|
| draft | Content Ops reasoning admin visibility | `extracted_content_pipeline/api/reasoning_contexts.py`, `tests/test_extracted_campaign_reasoning_context_api.py`, `extracted_content_pipeline/README.md`, `extracted_content_pipeline/STATUS.md` | codex-2026-05-15 | Avoid DB reasoning admin/API edits |

This table is for PRs we need to coordinate around, not a mirror of `gh pr list`. Use `gh pr list --state open` for the full inventory.

# In-Flight PRs

Last updated: 2026-05-05T19:20Z by codex-2026-05-05-d17

Add a row before opening a PR (session protocol step 2). Drop the row when the PR merges (step 4). See [`../COORDINATION.md`](../COORDINATION.md) for protocol details.

| PR | Title | Touches | Owner | Don't conflict with |
|---|---|---|---|---|
| TBA | D17 content visibility sink adapters | `extracted_content_pipeline/campaign_visibility.py`, campaign visibility tests, content pipeline README/runbook/status | codex-2026-05-05-d17 | Avoid editing AI Content Ops visibility sink adapter surfaces until this PR lands |

This table is for PRs we need to coordinate around, not a mirror of `gh pr list`. Use `gh pr list --state open` for the full inventory.

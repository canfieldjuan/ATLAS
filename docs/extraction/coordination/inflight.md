# In-Flight PRs

Last updated: 2026-05-19T18:54Z by codex-2026-05-19

Add a row before opening a PR (session protocol step 2). Drop the row when the PR merges (step 4). See [`../COORDINATION.md`](../COORDINATION.md) for protocol details.

| PR | Title | Touches | Owner | Don't conflict with |
|---|---|---|---|---|
| TBD | PR-Content-Selling-Context-Inputs | `extracted_content_pipeline/content_ops_execution.py`, `extracted_content_pipeline/campaign_generation.py`, `tests/test_extracted_content_ops_execution.py`, `tests/test_extracted_campaign_generation.py` | codex-2026-05-19 | Campaign execution dispatcher, campaign prompt opportunity metadata |

This table is for PRs we need to coordinate around, not a mirror of `gh pr list`. Use `gh pr list --state open` for the full inventory.

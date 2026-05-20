# In-Flight PRs

Last updated: 2026-05-20T17:08Z by codex-2026-05-20

Add a row before opening a PR (session protocol step 2). Drop the row when the PR merges (step 4). See [`../COORDINATION.md`](../COORDINATION.md) for protocol details.

| PR | Title | Touches | Owner | Don't conflict with |
|---|---|---|---|---|
| PR-Content-Ops-FAQ-Complaint-Source-Policy | Generic complaint source aliases and FAQ financial complaint actions | `extracted_content_pipeline/campaign_source_adapters.py`, `extracted_content_pipeline/ticket_faq_markdown.py`, `tests/test_extracted_campaign_source_adapters.py`, `tests/test_extracted_ticket_faq_markdown.py` | codex-2026-05-20 | open PR #691 blog publish metadata |

This table is for PRs we need to coordinate around, not a mirror of `gh pr list`. Use `gh pr list --state open` for the full inventory.

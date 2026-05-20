# In-Flight PRs

Last updated: 2026-05-20T01:13Z by codex-2026-05-20

Add a row before opening a PR (session protocol step 2). Drop the row when the PR merges (step 4). See [`../COORDINATION.md`](../COORDINATION.md) for protocol details.

| PR | Title | Touches | Owner | Don't conflict with |
|---|---|---|---|---|
| TBD | PR-Content-Ops-FAQ-Markdown-Render-Proof | `tests/test_extracted_ticket_faq_markdown.py`; FAQ Markdown render-proof test | codex-2026-05-20 | Avoid concurrent edits to the ticket FAQ Markdown tests. |

This table is for PRs we need to coordinate around, not a mirror of `gh pr list`. Use `gh pr list --state open` for the full inventory.

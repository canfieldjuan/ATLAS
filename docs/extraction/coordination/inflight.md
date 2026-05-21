# In-Flight PRs

Last updated: 2026-05-20T17:08Z by codex-2026-05-20

Add a row before opening a PR (session protocol step 2). Drop the row when the PR merges (step 4). See [`../COORDINATION.md`](../COORDINATION.md) for protocol details.

| PR | Title | Touches | Owner | Don't conflict with |
|---|---|---|---|---|
| PR-Content-Ops-FAQ-Source-Context-Policy | Use preserved source context for FAQ intent/action policy | `extracted_content_pipeline/ticket_faq_markdown.py`, `tests/test_extracted_ticket_faq_markdown.py`, `docs/audits/ai_content_ops_deferred_backlog_2026-05-11.md` | codex-2026-05-20 | blog PR #697 owns blog content only |

This table is for PRs we need to coordinate around, not a mirror of `gh pr list`. Use `gh pr list --state open` for the full inventory.

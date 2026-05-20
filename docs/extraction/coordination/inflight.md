# In-Flight PRs

Last updated: 2026-05-20T03:21Z by codex-2026-05-20

Add a row before opening a PR (session protocol step 2). Drop the row when the PR merges (step 4). See [`../COORDINATION.md`](../COORDINATION.md) for protocol details.

| PR | Title | Touches | Owner | Don't conflict with |
|---|---|---|---|---|
| TBD | PR-Content-Ops-FAQ-Customer-Wording | `plans/PR-Content-Ops-FAQ-Customer-Wording.md`; `docs/extraction/coordination/inflight.md`; `extracted_content_pipeline/ticket_faq_markdown.py`; `tests/test_extracted_ticket_faq_markdown.py` | codex-2026-05-20 | Avoid concurrent edits to FAQ Markdown question wording. |

This table is for PRs we need to coordinate around, not a mirror of `gh pr list`. Use `gh pr list --state open` for the full inventory.

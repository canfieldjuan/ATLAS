# In-Flight PRs

Last updated: 2026-05-19T23:45Z by codex-2026-05-19

Add a row before opening a PR (session protocol step 2). Drop the row when the PR merges (step 4). See [`../COORDINATION.md`](../COORDINATION.md) for protocol details.

| PR | Title | Touches | Owner | Don't conflict with |
|---|---|---|---|---|
| TBD | PR-Content-Ops-FAQ-Generated-Asset-Persistence | `ticket_faq_markdown.py`; `ticket_faq_ports.py`; `ticket_faq_postgres.py`; FAQ migration; host wiring/tests/docs | codex-2026-05-19 | Avoid concurrent edits to `faq_markdown` persistence and generated-asset storage seam. |

This table is for PRs we need to coordinate around, not a mirror of `gh pr list`. Use `gh pr list --state open` for the full inventory.

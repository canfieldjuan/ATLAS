# In-Flight PRs

Last updated: 2026-05-20T04:01Z by codex-2026-05-20

Add a row before opening a PR (session protocol step 2). Drop the row when the PR merges (step 4). See [`../COORDINATION.md`](../COORDINATION.md) for protocol details.

| PR | Title | Touches | Owner | Don't conflict with |
|---|---|---|---|---|
| TBD | PR-Content-Ops-FAQ-Output-Checks | `plans/PR-Content-Ops-FAQ-Output-Checks.md`; `docs/extraction/coordination/inflight.md`; `extracted_content_pipeline/ticket_faq_markdown.py`; `atlas-intel-ui/src/pages/ContentOpsAssetsReview.tsx`; `scripts/smoke_extracted_content_ops_execution.py`; `tests/test_extracted_content_ops_execution_smoke.py`; `tests/test_extracted_ticket_faq_markdown.py` | codex-2026-05-20 | Avoid concurrent edits to FAQ Markdown output-check semantics and generated-asset review labels. |

This table is for PRs we need to coordinate around, not a mirror of `gh pr list`. Use `gh pr list --state open` for the full inventory.

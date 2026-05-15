# In-Flight PRs

Last updated: 2026-05-15T18:58Z by codex-2026-05-15

Add a row before opening a PR (session protocol step 2). Drop the row when the PR merges (step 4). See [`../COORDINATION.md`](../COORDINATION.md) for protocol details.

| PR | Title | Touches | Owner | Don't conflict with |
|---|---|---|---|---|
| draft | Content Ops generated asset batch scale | `extracted_content_pipeline/api/generated_assets.py`, `extracted_content_pipeline/blog_post_postgres.py`, `extracted_content_pipeline/report_postgres.py`, `extracted_content_pipeline/landing_page_postgres.py`, `extracted_content_pipeline/sales_brief_postgres.py`, `extracted_content_pipeline/blog_ports.py`, `extracted_content_pipeline/report_ports.py`, `extracted_content_pipeline/landing_page_ports.py`, `extracted_content_pipeline/sales_brief_ports.py`, `tests/test_extracted_content_asset_api.py`, `docs/audits/ai_content_ops_deferred_backlog_2026-05-11.md`, `docs/extraction/coordination/inflight.md`, `plans/PR-Content-Ops-Generated-Asset-Batch-Scale.md` | codex-2026-05-15 | Avoid generated asset review backend and backlog docs |

This table is for PRs we need to coordinate around, not a mirror of `gh pr list`. Use `gh pr list --state open` for the full inventory.

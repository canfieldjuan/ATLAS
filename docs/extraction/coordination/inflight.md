# In-Flight PRs

Last updated: 2026-05-04T19:20Z by codex-content-analytics-worker

Add a row before opening a PR (session protocol step 2). Drop the row when the PR merges (step 4). See [`../COORDINATION.md`](../COORDINATION.md) for protocol details.

| PR | Title | Touches | Owner | Don't conflict with |
|---|---|---|---|---|
| #164 | docs: log cross-product standalone % audit | `docs/extraction/cross_product_audit_2026-05-04.md` | canfieldjuan | Avoid editing the cross-product audit doc until PR #164 lands |
| (branch: codex/content-pipeline-next-seam) | Add AI Content Ops analytics refresh worker CLI | `extracted_content_pipeline/campaign_postgres_analytics.py`; `scripts/refresh_extracted_campaign_analytics.py`; `tests/test_extracted_campaign_postgres_analytics.py`; content-pipeline docs/status/manifest/check wiring | codex-content-analytics-worker | Avoid content-pipeline analytics refresh runner/CLI/doc wiring until this branch lands |

This table is for PRs we need to coordinate around, not a mirror of `gh pr list`. Use `gh pr list --state open` for the full inventory.

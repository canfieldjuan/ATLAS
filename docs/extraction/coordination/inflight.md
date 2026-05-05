# In-Flight PRs

Last updated: 2026-05-05T02:23Z by codex-content-category-intel-worker

Add a row before opening a PR (session protocol step 2). Drop the row when the PR merges (step 4). See [`../COORDINATION.md`](../COORDINATION.md) for protocol details.

| PR | Title | Touches | Owner | Don't conflict with |
|---|---|---|---|---|
| #164 | docs: log cross-product standalone % audit | `docs/extraction/cross_product_audit_2026-05-04.md` | canfieldjuan | Avoid editing the cross-product audit doc until PR #164 lands |
| #209 | Add seller category intelligence refresh seam | NEW: `extracted_content_pipeline/campaign_postgres_seller_category_intelligence.py`; NEW: `scripts/refresh_extracted_seller_category_intelligence.py`; NEW: `tests/test_extracted_campaign_postgres_seller_category_intelligence.py`; EDIT: extracted content pipeline manifest/import smoke/checks/docs/status/coordination. No Atlas source edits. | codex-content-category-intel-worker | Avoid seller category intelligence refresh files until this PR lands |

This table is for PRs we need to coordinate around, not a mirror of `gh pr list`. Use `gh pr list --state open` for the full inventory.

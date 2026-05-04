# In-Flight PRs

Last updated: 2026-05-04T20:53Z by codex-content-api-webhook-worker

Add a row before opening a PR (session protocol step 2). Drop the row when the PR merges (step 4). See [`../COORDINATION.md`](../COORDINATION.md) for protocol details.

| PR | Title | Touches | Owner | Don't conflict with |
|---|---|---|---|---|
| TBD | Add AI Content Ops webhook API router factory | `extracted_content_pipeline/api/campaign_webhooks.py`, `tests/test_extracted_campaign_api_webhooks.py`, content pipeline docs/manifest/check wiring | codex-content-api-webhook-worker | Avoid editing extracted campaign webhook API factory seams until this lands |
| #164 | docs: log cross-product standalone % audit | `docs/extraction/cross_product_audit_2026-05-04.md` | canfieldjuan | Avoid editing the cross-product audit doc until PR #164 lands |

This table is for PRs we need to coordinate around, not a mirror of `gh pr list`. Use `gh pr list --state open` for the full inventory.

# In-Flight PRs

Last updated: 2026-05-05T05:51Z by codex-2026-05-05-hosted-ops

Add a row before opening a PR (session protocol step 2). Drop the row when the PR merges (step 4). See [`../COORDINATION.md`](../COORDINATION.md) for protocol details.

| PR | Title | Touches | Owner | Don't conflict with |
|---|---|---|---|---|
| #164 | docs: log cross-product standalone % audit | `docs/extraction/cross_product_audit_2026-05-04.md` | canfieldjuan | Avoid editing the cross-product audit doc until PR #164 lands |
| #233 | Add hosted seller operations API triggers | `extracted_content_pipeline/api/seller_campaigns.py`; `tests/test_extracted_campaign_api_seller_campaigns.py`; extracted content pipeline README/runbook/status/coordination docs. No Atlas source edits. | codex-2026-05-05-hosted-ops | Avoid seller campaign API refresh/prepare orchestration files until this PR lands |

This table is for PRs we need to coordinate around, not a mirror of `gh pr list`. Use `gh pr list --state open` for the full inventory.

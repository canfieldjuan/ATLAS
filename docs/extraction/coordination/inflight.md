# In-Flight PRs

Last updated: 2026-05-05T14:05Z by codex-2026-05-05-d9

Add a row before opening a PR (session protocol step 2). Drop the row when the PR merges (step 4). See [`../COORDINATION.md`](../COORDINATION.md) for protocol details.

| PR | Title | Touches | Owner | Don't conflict with |
|---|---|---|---|---|
| #164 | docs: log cross-product standalone % audit | `docs/extraction/cross_product_audit_2026-05-04.md` | canfieldjuan | Avoid editing the cross-product audit doc until PR #164 lands |
| TBD | D9 hosted campaign operations API triggers | `extracted_content_pipeline/api/campaign_operations.py`, `tests/test_extracted_campaign_api_operations.py`, `extracted_content_pipeline/manifest.json`, `scripts/run_extracted_pipeline_checks.sh`, content pipeline docs/status | codex-2026-05-05-d9 | Avoid adding campaign send/progression/analytics API triggers or editing the same extracted pipeline runner entry until this PR lands |

This table is for PRs we need to coordinate around, not a mirror of `gh pr list`. Use `gh pr list --state open` for the full inventory.

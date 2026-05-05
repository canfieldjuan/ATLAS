# In-Flight PRs

Last updated: 2026-05-05T02:51Z by codex-2026-05-05

Add a row before opening a PR (session protocol step 2). Drop the row when the PR merges (step 4). See [`../COORDINATION.md`](../COORDINATION.md) for protocol details.

| PR | Title | Touches | Owner | Don't conflict with |
|---|---|---|---|---|
| #164 | docs: log cross-product standalone % audit | `docs/extraction/cross_product_audit_2026-05-04.md` | canfieldjuan | Avoid editing the cross-product audit doc until PR #164 lands |
| #209 | Add seller category intelligence refresh seam | NEW: `extracted_content_pipeline/campaign_postgres_seller_category_intelligence.py`; NEW: `scripts/refresh_extracted_seller_category_intelligence.py`; NEW: `tests/test_extracted_campaign_postgres_seller_category_intelligence.py`; EDIT: extracted content pipeline manifest/import smoke/checks/docs/status/coordination. No Atlas source edits. | codex-2026-05-05 | Avoid seller category intelligence refresh files until this PR lands |
| (pending) | Route battle-card runtime helpers through support port | `atlas_brain/services/b2b/battle_card_ports.py`; `extracted_competitive_intelligence/services/b2b/battle_card_ports.py`; `atlas_brain/autonomous/tasks/b2b_battle_cards.py`; `extracted_competitive_intelligence/autonomous/tasks/b2b_battle_cards.py`; `tests/test_extracted_competitive_battle_card_ports.py`; `extracted_competitive_intelligence/README.md`; `extracted_competitive_intelligence/STATUS.md` | codex-2026-05-05 | Competitive battle-card churn-scope/progress seams only; avoid synthesis reader, reasoning core, LLM gateway, content pipeline, and cross-product audit files |

This table is for PRs we need to coordinate around, not a mirror of `gh pr list`. Use `gh pr list --state open` for the full inventory.

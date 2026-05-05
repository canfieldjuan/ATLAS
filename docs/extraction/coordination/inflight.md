# In-Flight PRs

Last updated: 2026-05-05T05:15Z by codex-2026-05-05

Add a row before opening a PR (session protocol step 2). Drop the row when the PR merges (step 4). See [`../COORDINATION.md`](../COORDINATION.md) for protocol details.

| PR | Title | Touches | Owner | Don't conflict with |
|---|---|---|---|---|
| #164 | docs: log cross-product standalone % audit | `docs/extraction/cross_product_audit_2026-05-04.md` | canfieldjuan | Avoid editing the cross-product audit doc until PR #164 lands |
| #247 | Route Competitive ProductClaim through extracted quality gate | `extracted_competitive_intelligence/services/b2b/product_claim.py`; `extracted_competitive_intelligence/manifest.json`; `tests/test_extracted_competitive_manifest.py`; `tests/test_extracted_competitive_product_claim.py`; `extracted_competitive_intelligence/README.md`; `extracted_competitive_intelligence/STATUS.md` | codex-2026-05-05 | Competitive ProductClaim compatibility only; avoid Atlas product-claim core, quality-gate implementation, battle-card/vendor-briefing task ports, content pipeline, and cross-product audit files |

This table is for PRs we need to coordinate around, not a mirror of `gh pr list`. Use `gh pr list --state open` for the full inventory.

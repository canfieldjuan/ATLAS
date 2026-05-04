# In-Flight PRs

Last updated: 2026-05-04T20:08Z by codex-content-webhook-worker

Add a row before opening a PR (session protocol step 2). Drop the row when the PR merges (step 4). See [`../COORDINATION.md`](../COORDINATION.md) for protocol details.

| PR | Title | Touches | Owner | Don't conflict with |
|---|---|---|---|---|
| pending | Add Competitive Intelligence CRM provider standalone port | `extracted_competitive_intelligence/_standalone/crm_provider.py`; `extracted_competitive_intelligence/services/crm_provider.py`; `extracted_competitive_intelligence/README.md`; `extracted_competitive_intelligence/STATUS.md`; `scripts/run_extracted_competitive_intelligence_checks.sh`; `scripts/smoke_extracted_competitive_intelligence_standalone.py`; `tests/test_extracted_competitive_crm_provider_port.py` | codex-2026-05-04 | Competitive Intelligence CRM provider seam; avoid vendor briefing gate CRM adapter files |
| #164 | docs: log cross-product standalone % audit | `docs/extraction/cross_product_audit_2026-05-04.md` | canfieldjuan | Avoid editing the cross-product audit doc until PR #164 lands |

This table is for PRs we need to coordinate around, not a mirror of `gh pr list`. Use `gh pr list --state open` for the full inventory.

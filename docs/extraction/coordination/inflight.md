# In-Flight PRs

Last updated: 2026-05-05T01:38Z by claude-2026-05-03

Add a row before opening a PR (session protocol step 2). Drop the row when the PR merges (step 4). See [`../COORDINATION.md`](../COORDINATION.md) for protocol details.

| PR | Title | Touches | Owner | Don't conflict with |
|---|---|---|---|---|
| #214 | Route Competitive LLM router through extracted infrastructure | `extracted_competitive_intelligence/services/llm_router.py`; `atlas_brain/autonomous/tasks/b2b_vendor_briefing.py`; `extracted_competitive_intelligence/autonomous/tasks/b2b_vendor_briefing.py`; `scripts/smoke_extracted_competitive_intelligence_standalone.py`; `scripts/run_extracted_competitive_intelligence_checks.sh`; `tests/test_extracted_competitive_llm_router_bridge.py`; `extracted_competitive_intelligence/README.md`; `extracted_competitive_intelligence/STATUS.md` | codex-2026-05-05 | Competitive LLM router boundary only; avoid reasoning core ports and cross-product audit files |
| #164 | docs: log cross-product standalone % audit | `docs/extraction/cross_product_audit_2026-05-04.md` | canfieldjuan | Avoid editing the cross-product audit doc until PR #164 lands |

This table is for PRs we need to coordinate around, not a mirror of `gh pr list`. Use `gh pr list --state open` for the full inventory.

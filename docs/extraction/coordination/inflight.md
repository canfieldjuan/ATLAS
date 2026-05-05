# In-Flight PRs

Last updated: 2026-05-05T14:30Z by codex-2026-05-05

Add a row before opening a PR (session protocol step 2). Drop the row when the PR merges (step 4). See [`../COORDINATION.md`](../COORDINATION.md) for protocol details.

| PR | Title | Touches | Owner | Don't conflict with |
|---|---|---|---|---|
| #164 | docs: log cross-product standalone % audit | `docs/extraction/cross_product_audit_2026-05-04.md` | canfieldjuan | Avoid editing the cross-product audit doc until PR #164 lands |
| #260 | Route vendor briefing LLM/cache through support port | `atlas_brain/autonomous/tasks/b2b_vendor_briefing.py`; `atlas_brain/services/b2b/vendor_briefing_ports.py`; `extracted_competitive_intelligence/autonomous/tasks/b2b_vendor_briefing.py`; `extracted_competitive_intelligence/services/b2b/vendor_briefing_ports.py`; `extracted_competitive_intelligence/services/b2b/cache_runner.py`; `extracted_content_pipeline/autonomous/tasks/b2b_vendor_briefing.py`; `extracted_content_pipeline/services/b2b/vendor_briefing_ports.py`; `extracted_content_pipeline/services/llm_router.py`; `extracted_content_pipeline/pipelines/llm.py`; `tests/test_b2b_vendor_briefing.py`; `tests/test_extracted_competitive_vendor_briefing_ports.py`; `tests/test_extracted_vendor_briefing_seams.py`; `tests/test_extracted_campaign_llm_bridge.py`; `extracted_competitive_intelligence/README.md`; `extracted_competitive_intelligence/STATUS.md` | codex-2026-05-05 | Vendor briefing LLM/cache/tracing support only; avoid reasoning-core wrapper files, battle-card ports, content-pipeline API/orchestration surfaces, and API extraction |

This table is for PRs we need to coordinate around, not a mirror of `gh pr list`. Use `gh pr list --state open` for the full inventory.

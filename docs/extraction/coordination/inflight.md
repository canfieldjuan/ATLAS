# In-Flight PRs

Last updated: 2026-05-05T03:55Z by claude-2026-05-03

Add a row before opening a PR (session protocol step 2). Drop the row when the PR merges (step 4). See [`../COORDINATION.md`](../COORDINATION.md) for protocol details.

| PR | Title | Touches | Owner | Don't conflict with |
|---|---|---|---|---|
| #164 | docs: log cross-product standalone % audit | `docs/extraction/cross_product_audit_2026-05-04.md` | canfieldjuan | Avoid editing the cross-product audit doc until PR #164 lands |
| #242 | Route vendor briefing synthesis reader through intelligence port | `atlas_brain/services/b2b/vendor_briefing_ports.py`; `extracted_competitive_intelligence/services/b2b/vendor_briefing_ports.py`; `atlas_brain/autonomous/tasks/b2b_vendor_briefing.py`; `extracted_competitive_intelligence/autonomous/tasks/b2b_vendor_briefing.py`; `tests/test_extracted_competitive_vendor_briefing_ports.py`; `extracted_competitive_intelligence/README.md`; `extracted_competitive_intelligence/STATUS.md` | codex-2026-05-05 | Competitive vendor-briefing synthesis-reader seam only; avoid LLM/reasoning core, battle-card ports, content pipeline, and cross-product audit files |

This table is for PRs we need to coordinate around, not a mirror of `gh pr list`. Use `gh pr list --state open` for the full inventory.

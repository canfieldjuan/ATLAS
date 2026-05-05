# In-Flight PRs

Last updated: 2026-05-05T18:34Z by codex-2026-05-05

Add a row before opening a PR (session protocol step 2). Drop the row when the PR merges (step 4). See [`../COORDINATION.md`](../COORDINATION.md) for protocol details.

| PR | Title | Touches | Owner | Don't conflict with |
|---|---|---|---|---|
| #164 | docs: log cross-product standalone % audit | `docs/extraction/cross_product_audit_2026-05-04.md` | canfieldjuan | Avoid editing the cross-product audit doc until PR #164 lands |
| pending | Centralize Competitive Intelligence vendor briefing state | `atlas_brain/autonomous/tasks/b2b_vendor_briefing.py`, `atlas_brain/services/b2b/vendor_briefing_repository.py`, `extracted_competitive_intelligence/autonomous/tasks/b2b_vendor_briefing.py`, `extracted_competitive_intelligence/services/b2b/vendor_briefing_repository.py`, `extracted_competitive_intelligence/manifest.json`, `extracted_competitive_intelligence/STATUS.md`, competitive repository tests | codex-2026-05-05 | Avoid vendor briefing delivery/HITL persistence and scheduled pending-approval state edits until this lands |

This table is for PRs we need to coordinate around, not a mirror of `gh pr list`. Use `gh pr list --state open` for the full inventory.

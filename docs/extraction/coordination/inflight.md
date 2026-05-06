# In-Flight PRs

Last updated: 2026-05-06T18:26Z by claude-2026-05-03

Add a row before opening a PR (session protocol step 2). Drop the row when the PR merges (step 4). See [`../COORDINATION.md`](../COORDINATION.md) for protocol details.

| PR | Title | Touches | Owner | Don't conflict with |
|---|---|---|---|---|
| (in flight) | Prune vestigial `_blog_matching` bridge from extracted_competitive_intelligence (Phase 3 progress) | DELETE: `extracted_competitive_intelligence/autonomous/tasks/_blog_matching.py` (23-LOC Phase 1 atlas_brain bridge with zero callers anywhere in the repo). Drops competitive_intelligence atlas_brain import count 13→12. No tests / consumers / manifest entries touched. | claude-2026-05-03 | `extracted_competitive_intelligence/autonomous/tasks/_blog_matching.py` |

This table is for PRs we need to coordinate around, not a mirror of `gh pr list`. Use `gh pr list --state open` for the full inventory.

# In-Flight PRs

Last updated: 2026-05-07T03:00Z by codex-2026-05-07-d33-cleanup

Add a row before opening a PR (session protocol step 2). Drop the row when the PR merges (step 4). See [`../COORDINATION.md`](../COORDINATION.md) for protocol details.

| PR | Title | Touches | Owner | Don't conflict with |
|---|---|---|---|---|
| (in flight) | AI Content Ops blog-post service-shaped adapter | NEW/EDIT: `extracted_content_pipeline/blog_ports.py`, `extracted_content_pipeline/blog_generation.py`, `extracted_content_pipeline/control_surfaces.py`, `extracted_content_pipeline/generation_plan.py`, `extracted_content_pipeline/content_ops_execution.py`, `tests/test_extracted_blog_generation.py`, docs/status/check runner as needed. | codex-2026-05-07-blog-service | Competitive-intelligence Phase 3 files; PR #357 execution seam files until that PR merges. |
| (in flight) | AI Content Ops execution seam for runnable control-surface plans | NEW/EDIT: `extracted_content_pipeline/content_ops_execution.py`, `extracted_content_pipeline/api/control_surfaces.py`, `tests/test_extracted_content_ops_execution.py`, `tests/test_extracted_content_control_surface_api.py`, docs/status/check runner as needed. | codex-2026-05-07-exec | Competitive-intelligence Phase 3 files; extracted reasoning-core migration files. |
| (in flight) | Promote `autonomous/visibility.py` from Phase 1 bridge to manifest-synced mirror (Phase 3 progress, 1/7 active bridges) | EDIT: `extracted_competitive_intelligence/manifest.json` (+ source/target mapping for `autonomous/visibility.py` AND `storage/migrations/246_pipeline_visibility.sql` per Codex P2). REPLACE: `extracted_competitive_intelligence/autonomous/visibility.py` (23-LOC bridge -> 344-LOC mirror copy of atlas_brain peer via `sync_extracted.sh`). NEW: `extracted_competitive_intelligence/storage/migrations/246_pipeline_visibility.sql` (mirror of atlas migration creating `artifact_attempts` / `enrichment_quarantines` / `pipeline_visibility_events` / `pipeline_visibility_reviews` so standalone deployments don't silently drop visibility writes). atlas_brain count 12->11. | claude-2026-05-03 | `extracted_competitive_intelligence/manifest.json`; `extracted_competitive_intelligence/autonomous/visibility.py`; `extracted_competitive_intelligence/storage/migrations/246_pipeline_visibility.sql` |

This table is for PRs we need to coordinate around, not a mirror of `gh pr list`. Use `gh pr list --state open` for the full inventory.

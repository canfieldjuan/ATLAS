# In-Flight PRs

Last updated: 2026-05-06T19:20Z by codex-2026-05-06-d22

Add a row before opening a PR (session protocol step 2). Drop the row when the PR merges (step 4). See [`../COORDINATION.md`](../COORDINATION.md) for protocol details.

| PR | Title | Touches | Owner | Don't conflict with |
|---|---|---|---|---|
| (in flight) | Promote `autonomous/visibility.py` from Phase 1 bridge to manifest-synced mirror (Phase 3 progress, 1/7 active bridges) | EDIT: `extracted_competitive_intelligence/manifest.json` (+ source/target mapping for `autonomous/visibility.py` AND `storage/migrations/246_pipeline_visibility.sql` per Codex P2). REPLACE: `extracted_competitive_intelligence/autonomous/visibility.py` (23-LOC bridge -> 344-LOC mirror copy of atlas_brain peer via `sync_extracted.sh`). NEW: `extracted_competitive_intelligence/storage/migrations/246_pipeline_visibility.sql` (mirror of atlas migration creating `artifact_attempts` / `enrichment_quarantines` / `pipeline_visibility_events` / `pipeline_visibility_reviews` so standalone deployments don't silently drop visibility writes). atlas_brain count 12->11. | claude-2026-05-03 | `extracted_competitive_intelligence/manifest.json`; `extracted_competitive_intelligence/autonomous/visibility.py`; `extracted_competitive_intelligence/storage/migrations/246_pipeline_visibility.sql` |
| (in flight) | PR-D22: AI Content Ops host smoke command | `scripts/smoke_extracted_content_pipeline_host.py`, extracted content docs/tests/check runner | codex-2026-05-06-d22 | Avoid editing competitive-intelligence Phase 3 visibility files or cross-product audit docs. |

This table is for PRs we need to coordinate around, not a mirror of `gh pr list`. Use `gh pr list --state open` for the full inventory.

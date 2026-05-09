# In-Flight PRs

Last updated: 2026-05-09T00:00Z by codex-content-ops-reasoning-audit-ui-cleanup

Add a row before opening a PR (session protocol step 2). Drop the row when the PR merges (step 4). See [`../COORDINATION.md`](../COORDINATION.md) for protocol details.

| PR | Title | Touches | Owner | Don't conflict with |
|---|---|---|---|---|
| (in flight) | Surface actual reasoning usage in Content Ops step audit | EDIT: `extracted_content_pipeline/content_ops_execution.py`, `tests/test_extracted_content_ops_execution.py`, `atlas-intel-ui/src/api/contentOps.ts`, `atlas-intel-ui/src/domain/contentOps/{types.ts,fromWire.ts}`, `atlas-intel-ui/src/pages/ContentOpsNewRun.tsx`, `atlas-intel-ui/src/api/__fixtures__/contentOps/execution-completed.json`, `docs/frontend/content_ops_frontend_contract.md`; NEW: `plans/PR-Content-Ops-Reasoning-Usage-Audit.md`. | codex-content-ops-reasoning-usage-audit | Avoid concurrent edits to Content Ops execution reasoning audit shape and execution result UI badge. |
| (in flight) | Promote `autonomous/visibility.py` from Phase 1 bridge to manifest-synced mirror (Phase 3 progress, 1/7 active bridges) | EDIT: `extracted_competitive_intelligence/manifest.json` (+ source/target mapping for `autonomous/visibility.py` AND `storage/migrations/246_pipeline_visibility.sql` per Codex P2). REPLACE: `extracted_competitive_intelligence/autonomous/visibility.py` (23-LOC bridge -> 344-LOC mirror copy of atlas_brain peer via `sync_extracted.sh`). NEW: `extracted_competitive_intelligence/storage/migrations/246_pipeline_visibility.sql` (mirror of atlas migration creating `artifact_attempts` / `enrichment_quarantines` / `pipeline_visibility_events` / `pipeline_visibility_reviews` so standalone deployments don't silently drop visibility writes). atlas_brain count 12->11. | claude-2026-05-03 | `extracted_competitive_intelligence/manifest.json`; `extracted_competitive_intelligence/autonomous/visibility.py`; `extracted_competitive_intelligence/storage/migrations/246_pipeline_visibility.sql` |

This table is for PRs we need to coordinate around, not a mirror of `gh pr list`. Use `gh pr list --state open` for the full inventory.

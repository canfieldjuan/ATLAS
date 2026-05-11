# In-Flight PRs

Last updated: 2026-05-11T00:34Z by codex-content-ops-reasoning-ui-parity

Add a row before opening a PR (session protocol step 2). Drop the row when the PR merges (step 4). See [`../COORDINATION.md`](../COORDINATION.md) for protocol details.

| PR | Title | Touches | Owner | Don't conflict with |
|---|---|---|---|---|
| (in flight) | Promote `autonomous/visibility.py` from Phase 1 bridge to manifest-synced mirror (Phase 3 progress, 1/7 active bridges) | EDIT: `extracted_competitive_intelligence/manifest.json` (+ source/target mapping for `autonomous/visibility.py` AND `storage/migrations/246_pipeline_visibility.sql` per Codex P2). REPLACE: `extracted_competitive_intelligence/autonomous/visibility.py` (23-LOC bridge -> 344-LOC mirror copy of atlas_brain peer via `sync_extracted.sh`). NEW: `extracted_competitive_intelligence/storage/migrations/246_pipeline_visibility.sql` (mirror of atlas migration creating `artifact_attempts` / `enrichment_quarantines` / `pipeline_visibility_events` / `pipeline_visibility_reviews` so standalone deployments don't silently drop visibility writes). atlas_brain count 12->11. | claude-2026-05-03 | `extracted_competitive_intelligence/manifest.json`; `extracted_competitive_intelligence/autonomous/visibility.py`; `extracted_competitive_intelligence/storage/migrations/246_pipeline_visibility.sql` |
| (in flight) | Render Content Ops reasoning provider source and consumed contexts in Atlas Intel UI | NEW: `plans/PR-Content-Ops-Reasoning-UI-Parity.md`. EDIT: `atlas-intel-ui/src/api/contentOps.ts`; `atlas-intel-ui/src/domain/contentOps/types.ts`; `atlas-intel-ui/src/domain/contentOps/fromWire.ts`; `atlas-intel-ui/src/domain/contentOps/index.ts`; `atlas-intel-ui/src/pages/ContentOpsNewRun.tsx`; `atlas-intel-ui/src/api/__fixtures__/contentOps/catalog.json`; `atlas-intel-ui/src/api/__fixtures__/contentOps/execution-completed.json`; `docs/extraction/coordination/inflight.md`. | codex-content-ops-reasoning-ui-parity | `extracted_competitive_intelligence/manifest.json`; `extracted_competitive_intelligence/autonomous/visibility.py`; `extracted_competitive_intelligence/storage/migrations/246_pipeline_visibility.sql` |

This table is for PRs we need to coordinate around, not a mirror of `gh pr list`. Use `gh pr list --state open` for the full inventory.

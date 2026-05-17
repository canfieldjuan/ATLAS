# In-Flight PRs

Last updated: 2026-05-17T22:12Z by codex-2026-05-17

Add a row before opening a PR (session protocol step 2). Drop the row when the PR merges (step 4). See [`../COORDINATION.md`](../COORDINATION.md) for protocol details.

| PR | Title | Touches | Owner | Don't conflict with |
|---|---|---|---|---|
| #579 | Content Ops ingestion inspect API | `extracted_content_pipeline/api/control_surfaces.py`, `extracted_content_pipeline/ingestion_diagnostics.py`, `tests/test_extracted_content_control_surface_api.py`, `extracted_content_pipeline/docs/control_surface_preview_api.md`, `extracted_content_pipeline/STATUS.md`, `docs/extraction/coordination/inflight.md`, `docs/extraction/coordination/state.md`, `plans/PR-Content-Ops-Ingestion-Inspect-API.md` | codex-2026-05-17 | Avoid editing Content Ops control-surface API or ingestion diagnostics |

This table is for PRs we need to coordinate around, not a mirror of `gh pr list`. Use `gh pr list --state open` for the full inventory.

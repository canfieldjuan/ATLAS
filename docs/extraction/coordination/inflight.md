# In-Flight PRs

Last updated: 2026-05-18T02:13Z by codex-2026-05-17

Add a row before opening a PR (session protocol step 2). Drop the row when the PR merges (step 4). See [`../COORDINATION.md`](../COORDINATION.md) for protocol details.

| PR | Title | Touches | Owner | Don't conflict with |
|---|---|---|---|---|
| #582 | Content Ops ingestion import API | `extracted_content_pipeline/api/control_surfaces.py`, `tests/test_extracted_content_control_surface_api.py`, `extracted_content_pipeline/docs/control_surface_preview_api.md`, `extracted_content_pipeline/STATUS.md`, `docs/extraction/coordination/inflight.md`, `docs/extraction/coordination/state.md`, `plans/PR-Content-Ops-Ingestion-Import-API.md` | codex-2026-05-17 | Avoid editing Content Ops control-surface ingestion API routes |

This table is for PRs we need to coordinate around, not a mirror of `gh pr list`. Use `gh pr list --state open` for the full inventory.

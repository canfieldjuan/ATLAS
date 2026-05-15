# In-Flight PRs

Last updated: 2026-05-15T22:52Z by codex-2026-05-15

Add a row before opening a PR (session protocol step 2). Drop the row when the PR merges (step 4). See [`../COORDINATION.md`](../COORDINATION.md) for protocol details.

| PR | Title | Touches | Owner | Don't conflict with |
|---|---|---|---|---|
| draft | Content Ops reasoning status capabilities | `extracted_content_pipeline/api/control_surfaces.py`, `tests/test_extracted_content_control_surface_api.py`, `extracted_content_pipeline/docs/control_surface_preview_api.md`, `docs/extraction/coordination/inflight.md`, `plans/PR-Content-Ops-Reasoning-Status-Capabilities.md` | codex-2026-05-15 | Avoid control-surface reasoning status API/docs |

This table is for PRs we need to coordinate around, not a mirror of `gh pr list`. Use `gh pr list --state open` for the full inventory.

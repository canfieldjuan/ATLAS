# In-Flight PRs

Last updated: 2026-05-16T18:15Z by codex-2026-05-16

Add a row before opening a PR (session protocol step 2). Drop the row when the PR merges (step 4). See [`../COORDINATION.md`](../COORDINATION.md) for protocol details.

| PR | Title | Touches | Owner | Don't conflict with |
|---|---|---|---|---|
| #555 | Content Ops report/sales reasoning presets | `extracted_content_pipeline/control_surfaces.py`, `extracted_content_pipeline/generation_plan.py`, `extracted_content_pipeline/content_ops_execution.py`, `extracted_content_pipeline/api/control_surfaces.py`, `extracted_content_pipeline/manifest.json`, `tests/test_extracted_content_control_surface_api.py`, `tests/test_extracted_content_generation_plan.py`, `tests/test_extracted_content_ops_execution.py`, `plans/PR-Content-Ops-Report-Sales-Reasoning-Presets.md` | codex-2026-05-16 | Avoid editing Content Ops control-surface reasoning preset wiring until this lands. |

This table is for PRs we need to coordinate around, not a mirror of `gh pr list`. Use `gh pr list --state open` for the full inventory.

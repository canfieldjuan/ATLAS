# In-Flight PRs

Last updated: 2026-05-16T18:45Z by codex-2026-05-16

Add a row before opening a PR (session protocol step 2). Drop the row when the PR merges (step 4). See [`../COORDINATION.md`](../COORDINATION.md) for protocol details.

| PR | Title | Touches | Owner | Don't conflict with |
|---|---|---|---|---|
| #556 | Content Ops strict reasoning output policy | `extracted_content_pipeline/api/control_surfaces.py`, `extracted_content_pipeline/generation_plan.py`, `extracted_content_pipeline/reasoning_policy.py`, `tests/test_extracted_content_control_surface_api.py`, `tests/test_extracted_content_generation_plan.py`, `tests/test_extracted_content_reasoning_policy.py`, `tests/test_extracted_report_generation.py`, `tests/test_extracted_sales_brief_generation.py`, `plans/PR-Content-Ops-Strict-Reasoning-Policy.md` | codex-2026-05-16 | Avoid report/sales packaged reasoning preset runtime policy edits. |

This table is for PRs we need to coordinate around, not a mirror of `gh pr list`. Use `gh pr list --state open` for the full inventory.

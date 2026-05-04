# In-Flight PRs

Last updated: 2026-05-04T02:05Z by claude-2026-05-03

Add a row before opening a PR (session protocol step 2). Drop the row when the PR merges (step 4). See [`../COORDINATION.md`](../COORDINATION.md) for protocol details.

| PR | Title | Touches | Owner | Don't conflict with |
|---|---|---|---|---|
| (PR-C1j, in flight) | PR-C1j: Route `extracted_content_pipeline/reasoning/temporal.py` through reasoning core wrapper | EDIT: `extracted_content_pipeline/reasoning/temporal.py` (drop ~466-line drifted fork; replace with thin re-export wrapper from `extracted_reasoning_core.temporal` + `extracted_reasoning_core.types`). All temporal types/constants/`TemporalEngine` were already promoted to core in PR-C1b/PR-C1c, so no drift-forward needed. Existing `tests/test_extracted_reasoning_temporal.py` keeps green against the wrapper. | claude-2026-05-03 | `extracted_content_pipeline/reasoning/temporal.py`; `tests/test_extracted_reasoning_temporal.py` |

This table is for PRs we need to coordinate around, not a mirror of `gh pr list`. Use `gh pr list --state open` for the full inventory.

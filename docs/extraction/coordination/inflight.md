# In-Flight PRs

Last updated: 2026-05-03T22:11Z by claude-2026-05-03

Add a row before opening a PR (session protocol step 2). Drop the row when the PR merges (step 4). See [`../COORDINATION.md`](../COORDINATION.md) for protocol details.

| PR | Title | Touches | Owner | Don't conflict with |
|---|---|---|---|---|
| (PR-A1.5, queued) | Apply Copilot fixes that missed PR #87 merge | `extracted_llm_infrastructure/{skills/__init__.py, _standalone/config.py, STATUS.md}`; `scripts/smoke_extracted_llm_infrastructure_imports.py`; `scripts/smoke_extracted_llm_infrastructure_standalone.py` | claude-2026-05-03-b | the 5 files listed; opening immediately after PR-A2 |
| (PR-C1b, in flight) | Temporal consolidation -> reasoning core | NEW: `extracted_reasoning_core/temporal.py` (atlas canonical + content_pipeline's `_numeric_value` / `_row_get` defensive helpers + parameterized `MIN_DAYS_FOR_PERCENTILES` constructor arg). NEW: `tests/test_extracted_reasoning_core_temporal.py` (smoke). `extracted_content_pipeline/reasoning/temporal.py` -> wrapper conversion is a separate atomic PR (PR-C1j). | claude-2026-05-03 | `extracted_reasoning_core/temporal.py`; the content_pipeline temporal fork (we read it but do not edit in this PR); the atlas-side temporal stays untouched |

This table is for PRs we need to coordinate around, not a mirror of `gh pr list`. Use `gh pr list --state open` for the full inventory.

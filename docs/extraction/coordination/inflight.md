# In-Flight PRs

Last updated: 2026-05-04T00:53Z by codex-2026-05-03

Add a row before opening a PR (session protocol step 2). Drop the row when the PR merges (step 4). See [`../COORDINATION.md`](../COORDINATION.md) for protocol details.

| PR | Title | Touches | Owner | Don't conflict with |
|---|---|---|---|---|
| (PR-C1h, in flight) | PR-C1h: Route `extracted_content_pipeline/reasoning/archetypes.py` through reasoning core wrapper | EDIT: `extracted_content_pipeline/reasoning/archetypes.py` (drop ~590-line drifted fork; replace with thin re-export wrapper from `extracted_reasoning_core.archetypes`). Existing `tests/test_extracted_reasoning_archetypes.py` keeps green against the wrapper. | claude-2026-05-03 | `extracted_content_pipeline/reasoning/archetypes.py`; `tests/test_extracted_reasoning_archetypes.py` |
| #116 | Add AI Content Ops draft export path (PR-D8) | `extracted_content_pipeline/campaign_postgres.py`; `extracted_content_pipeline/campaign_postgres_export.py`; `scripts/export_extracted_campaign_drafts.py`; content-pipeline docs/status/manifest; focused export tests | codex-2026-05-03 | Do not touch `extracted_reasoning_core/**`, LLM-infra files, or copied Atlas task mirrors |

This table is for PRs we need to coordinate around, not a mirror of `gh pr list`. Use `gh pr list --state open` for the full inventory.

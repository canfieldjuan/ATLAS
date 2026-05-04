# In-Flight PRs

Last updated: 2026-05-04T04:45Z by codex-2026-05-04-content

Add a row before opening a PR (session protocol step 2). Drop the row when the PR merges (step 4). See [`../COORDINATION.md`](../COORDINATION.md) for protocol details.

| PR | Title | Touches | Owner | Don't conflict with |
|---|---|---|---|---|
| (PR-C1j, in flight) | PR-C1j: Route `extracted_content_pipeline/reasoning/temporal.py` through reasoning core wrapper | EDIT: `extracted_content_pipeline/reasoning/temporal.py` (drop ~466-line drifted fork; replace with thin re-export wrapper from `extracted_reasoning_core.temporal` + `extracted_reasoning_core.types`). All temporal types/constants/`TemporalEngine` were already promoted to core in PR-C1b/PR-C1c, so no drift-forward needed. Existing `tests/test_extracted_reasoning_temporal.py` keeps green against the wrapper. | claude-2026-05-03 | `extracted_content_pipeline/reasoning/temporal.py`; `tests/test_extracted_reasoning_temporal.py` |
| (PR #129, in flight) | Own competitive intelligence write/source impact surfaces | EDIT: `extracted_competitive_intelligence/{README.md,STATUS.md,manifest.json}`. EDIT: `extracted_competitive_intelligence/mcp/b2b/write_intelligence.py` and NEW `write_ports.py`. EDIT: `extracted_competitive_intelligence/services/scraping/capabilities.py`. EDIT: competitive extraction checks/workflow and NEW `tests/test_extracted_competitive_manifest.py`. | codex-2026-05-04 | `extracted_competitive_intelligence/**`; `scripts/run_extracted_competitive_intelligence_checks.sh`; `scripts/smoke_extracted_competitive_intelligence_standalone.py`; `.github/workflows/extracted_competitive_intelligence_checks.yml`; `tests/test_extracted_competitive_manifest.py` |
| (PR-D10, in flight) | Add AI Content Ops queued send worker CLI | NEW: `extracted_content_pipeline/campaign_postgres_send.py`; NEW: `scripts/send_extracted_campaigns.py`; NEW: `tests/test_extracted_campaign_postgres_send.py`. EDIT: `extracted_content_pipeline/{README.md,STATUS.md,manifest.json}`; `extracted_content_pipeline/docs/{host_install_runbook.md,standalone_productization.md}`; `scripts/run_extracted_pipeline_checks.sh`; `tests/test_extracted_campaign_manifest.py`. | codex-2026-05-04-content | `extracted_content_pipeline/campaign_postgres_send.py`; `scripts/send_extracted_campaigns.py`; `tests/test_extracted_campaign_postgres_send.py`; listed content-pipeline docs and manifest files |

This table is for PRs we need to coordinate around, not a mirror of `gh pr list`. Use `gh pr list --state open` for the full inventory.

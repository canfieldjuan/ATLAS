# In-Flight PRs

Last updated: 2026-05-18T20:17Z by codex-2026-05-18

Add a row before opening a PR (session protocol step 2). Drop the row when the PR merges (step 4). See [`../COORDINATION.md`](../COORDINATION.md) for protocol details.

| PR | Title | Touches | Owner | Don't conflict with |
|---|---|---|---|---|
| pending | Content Ops source-row target defaults | `extracted_content_pipeline/campaign_source_adapters.py`, `extracted_content_pipeline/ingestion_diagnostics.py`, `extracted_content_pipeline/api/control_surfaces.py`, `scripts/build_extracted_campaign_opportunities_from_sources.py`, `scripts/inspect_extracted_content_ingestion.py`, `scripts/load_extracted_campaign_opportunities.py`, `scripts/run_extracted_campaign_generation_example.py`, `scripts/smoke_extracted_content_pipeline_host.py`, `tests/test_extracted_campaign_source_adapters.py`, `tests/test_extracted_campaign_generation_example.py`, `tests/test_extracted_content_ingestion_diagnostics.py`, `tests/test_extracted_content_control_surface_api.py`, `extracted_content_pipeline/README.md`, `extracted_content_pipeline/docs/host_install_runbook.md`, `extracted_content_pipeline/STATUS.md`, `docs/extraction/coordination/inflight.md`, `docs/extraction/coordination/state.md`, `plans/PR-Content-Ops-Source-Target-Defaults.md` | codex-2026-05-18 | Avoid editing Content Ops source-row ingestion defaults or ingestion docs |

This table is for PRs we need to coordinate around, not a mirror of `gh pr list`. Use `gh pr list --state open` for the full inventory.

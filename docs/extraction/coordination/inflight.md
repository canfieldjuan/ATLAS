# In-Flight PRs

Last updated: 2026-05-19T02:35Z by codex-2026-05-18

Add a row before opening a PR (session protocol step 2). Drop the row when the PR merges (step 4). See [`../COORDINATION.md`](../COORDINATION.md) for protocol details.

| PR | Title | Touches | Owner | Don't conflict with |
|---|---|---|---|---|
| TBD | PR-Content-Ops-CFPB-Source | `scripts/export_content_ops_cfpb_sources.py`, `scripts/smoke_content_ops_cfpb_source_postgres.py`, `scripts/run_extracted_pipeline_checks.sh`, `extracted_content_pipeline/campaign_example.py`, `tests/test_export_content_ops_cfpb_sources.py`, `tests/test_smoke_content_ops_cfpb_source_postgres.py`, `tests/test_extracted_campaign_generation_example.py`, `extracted_content_pipeline/README.md`, `extracted_content_pipeline/docs/host_install_runbook.md`, `plans/PR-Content-Ops-CFPB-Source.md` | codex-2026-05-18 | Content Ops source ingestion scripts/docs |

This table is for PRs we need to coordinate around, not a mirror of `gh pr list`. Use `gh pr list --state open` for the full inventory.

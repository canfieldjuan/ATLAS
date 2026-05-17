# In-Flight PRs

Last updated: 2026-05-17T21:52Z by codex-2026-05-17

Add a row before opening a PR (session protocol step 2). Drop the row when the PR merges (step 4). See [`../COORDINATION.md`](../COORDINATION.md) for protocol details.

| PR | Title | Touches | Owner | Don't conflict with |
|---|---|---|---|---|
| #578 | Content Ops ingestion diagnostics | `extracted_content_pipeline/ingestion_diagnostics.py`, `scripts/inspect_extracted_content_ingestion.py`, `tests/test_extracted_content_ingestion_diagnostics.py`, `scripts/run_extracted_pipeline_checks.sh`, `extracted_content_pipeline/README.md`, `extracted_content_pipeline/docs/host_install_runbook.md`, `docs/extraction/coordination/inflight.md`, `docs/extraction/coordination/state.md`, `plans/PR-Content-Ops-Ingestion-Diagnostics.md` | codex-2026-05-17 | Avoid editing Content Ops ingestion diagnostics, source/import docs, or extracted pipeline checks |

This table is for PRs we need to coordinate around, not a mirror of `gh pr list`. Use `gh pr list --state open` for the full inventory.

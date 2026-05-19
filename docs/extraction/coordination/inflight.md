# In-Flight PRs

Last updated: 2026-05-19T04:11Z by codex-2026-05-18

Add a row before opening a PR (session protocol step 2). Drop the row when the PR merges (step 4). See [`../COORDINATION.md`](../COORDINATION.md) for protocol details.

| PR | Title | Touches | Owner | Don't conflict with |
|---|---|---|---|---|
| TBD | PR-Content-Ops-Source-File-Postgres-Smoke | `scripts/smoke_content_ops_source_file_postgres.py`, `scripts/run_extracted_pipeline_checks.sh`, `tests/test_smoke_content_ops_source_file_postgres.py`, `extracted_content_pipeline/README.md`, `extracted_content_pipeline/docs/host_install_runbook.md`, `docs/extraction/coordination/inflight.md`, `plans/PR-Content-Ops-Source-File-Postgres-Smoke.md` | codex-2026-05-18 | Content Ops source-file Postgres smoke |

This table is for PRs we need to coordinate around, not a mirror of `gh pr list`. Use `gh pr list --state open` for the full inventory.

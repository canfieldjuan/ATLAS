# In-Flight PRs

Last updated: 2026-05-14T18:55Z by codex-2026-05-14-blog-blueprint-ingestion

Add a row before opening a PR (session protocol step 2). Drop the row when the PR merges (step 4). See [`../COORDINATION.md`](../COORDINATION.md) for protocol details.

| PR | Title | Touches | Owner | Don't conflict with |
|---|---|---|---|---|
| (in flight) | Add Content Ops blog blueprint ingestion CLI | NEW: `plans/PR-Content-Ops-Blog-Blueprint-Ingestion.md`; `extracted_content_pipeline/blog_blueprint_ingest.py`; `scripts/load_extracted_blog_blueprints.py`; `tests/test_extracted_blog_blueprint_ingest.py`. EDIT: `scripts/run_extracted_pipeline_checks.sh`; `extracted_content_pipeline/manifest.json`; `extracted_content_pipeline/README.md`; `extracted_content_pipeline/STATUS.md`; `extracted_content_pipeline/docs/host_install_runbook.md`; `extracted_content_pipeline/docs/standalone_productization.md`; `docs/extraction/coordination/inflight.md`. | codex-2026-05-14-blog-blueprint-ingestion | reasoning-provider backlog slices; frontend Content Ops UI slices |

This table is for PRs we need to coordinate around, not a mirror of `gh pr list`. Use `gh pr list --state open` for the full inventory.

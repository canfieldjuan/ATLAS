# In-Flight PRs

Last updated: 2026-05-19T03:43Z by codex-2026-05-18

Add a row before opening a PR (session protocol step 2). Drop the row when the PR merges (step 4). See [`../COORDINATION.md`](../COORDINATION.md) for protocol details.

| PR | Title | Touches | Owner | Don't conflict with |
|---|---|---|---|---|
| TBD | PR-Content-Ops-Source-Smoke-Shared-Helpers | `scripts/content_ops_source_postgres_smoke_helpers.py`, `scripts/smoke_content_ops_review_source_postgres.py`, `scripts/smoke_content_ops_cfpb_source_postgres.py`, `scripts/run_extracted_pipeline_checks.sh`, `tests/test_content_ops_source_postgres_smoke_helpers.py`, `tests/test_smoke_content_ops_review_source_postgres.py`, `docs/extraction/coordination/inflight.md`, `plans/PR-Content-Ops-Source-Smoke-Shared-Helpers.md` | codex-2026-05-18 | Content Ops source Postgres smoke helpers |

This table is for PRs we need to coordinate around, not a mirror of `gh pr list`. Use `gh pr list --state open` for the full inventory.

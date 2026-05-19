# In-Flight PRs

Last updated: 2026-05-19T01:30Z by codex-2026-05-18

Add a row before opening a PR (session protocol step 2). Drop the row when the PR merges (step 4). See [`../COORDINATION.md`](../COORDINATION.md) for protocol details.

| PR | Title | Touches | Owner | Don't conflict with |
|---|---|---|---|---|
| pending | Content Ops review-source schema preflight | `scripts/smoke_content_ops_review_source_postgres.py`, `tests/test_smoke_content_ops_review_source_postgres.py`, `extracted_content_pipeline/README.md`, `extracted_content_pipeline/docs/host_install_runbook.md`, `extracted_content_pipeline/STATUS.md`, `docs/extraction/coordination/inflight.md`, `docs/extraction/coordination/state.md`, `plans/PR-Content-Ops-Review-Source-Schema-Preflight.md` | codex-2026-05-18 | Avoid editing the review-source Postgres smoke script, host runbook review-source DB smoke docs, or related tests |

This table is for PRs we need to coordinate around, not a mirror of `gh pr list`. Use `gh pr list --state open` for the full inventory.
